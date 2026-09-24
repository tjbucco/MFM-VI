"""Baseline SVI mixtures: DP (stick-breaking) and fixed-T finite mixture.

The SVI core is agnostic to both the component family and the weight model.
Concrete classes are composed at the bottom of this file:

    atom family   x   weight model
    ------------      ------------
    BetaBernoulliAtoms    DPWeights      -> DPMixtureSVI
    NormalWishartAtoms    FiniteWeights  -> FiniteMixtureSVIGaussian, ...
    StudentTAtoms

An atom family supplies: _setup_atoms, _atom_init, _expected_log_lik,
_blend_atoms, _atom_stats, _kl_atoms, _predict_log_lik, _atom_params,
_invalidate_atoms.  A weight model supplies: __init__, _init_weights_prior,
_set_weights_from_counts, _E_log_weights, _blend_weights, _kl_weights,
_weight_state, mixture_weights, posterior_summaries.

Model:  y_i | z_i = k ~ f(y | theta_k),  theta_k ~ G0,  z_i ~ Cat(weights)

Initialisation
--------------
Atoms are initialised from `init_data` if given (else a subsample of Y of
size `init_size`), with pseudo-counts scaled by `init_scale` (default N /
n_init so the init is commensurate with the N/B-scaled minibatch updates).
For streaming, pass init_data = warm-up buffer and let the scale default to
N / n_warm; an unscaled init is overwritten by the first few batches.

Fitting
-------
fit() draws random minibatches unless `batch_order` (a sequence of index
arrays, one per iteration) is given, in which case batches are processed in
that order -- a single pass in arrival order is a stream.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import time
import numpy as np
from scipy.special import psi, logsumexp, gammaln

from CAVI.component_fams.betabernoulli_atoms import BetaBernoulliAtoms, sample_kplus
from CAVI.component_fams.gaussian_atoms import NormalWishartAtoms
from CAVI.component_fams.studentt_atoms import StudentTAtoms
from CAVI.component_fams.wishart_atoms import ZeroMeanWishartAtoms


class _MixtureSVICore:
    _tag = "SVI"

    # =================================================================
    # Setup / initialisation
    # =================================================================
    def _setup(self, Y, T, init, init_size, random_state,
               init_data=None, init_scale=None, **atom_kw):
        self.Y = np.asarray(Y, dtype=float)
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        self.N, self.p = self.Y.shape
        self.T = int(T)
        self.init, self.init_size = init, init_size
        self.init_data = None if init_data is None else np.asarray(init_data, float)
        self.init_scale = init_scale
        self.random_state = random_state
        self._setup_atoms(**atom_kw)

    def _init_atoms(self, random_state):
        """Returns (Y0, scale): the data the atoms were initialised on and
        the pseudo-count scale applied to it."""
        if self.init_data is not None:
            Y0 = self.init_data
        else:
            rng = np.random.default_rng(random_state)
            if self.init_size is None or self.init_size >= self.N:
                Y0 = self.Y
            else:
                Y0 = self.Y[rng.choice(self.N, size=self.init_size, replace=False)]
        scale = self.N / len(Y0) if self.init_scale is None else float(self.init_scale)
        self._atom_init(Y0, scale, random_state, mode=self.init)
        return Y0, scale

    def _full_initialize(self, random_state):
        self._init_weights_prior()
        Y0, scale = self._init_atoms(random_state)
        r0 = self._responsibilities(self._log_rho(Y0))
        self._set_weights_from_counts(scale * r0.sum(axis=0))
        self.r = None
        self.elbo_history = []

    # =================================================================
    # Local step
    # =================================================================
    def _log_rho(self, Yb):
        return self._E_log_weights()[None, :] + self._expected_log_lik(Yb)

    @staticmethod
    def _responsibilities(log_rho):
        return np.exp(log_rho - logsumexp(log_rho, axis=1, keepdims=True))

    def _compute_full_responsibilities(self):
        self.r = self._responsibilities(self._log_rho(self.Y))
        return self.r

    # =================================================================
    # State
    # =================================================================
    def _save_state(self):
        state = {k: np.copy(getattr(self, k)) for k in self._atom_params}
        state["elbo_history"] = list(self.elbo_history)
        state.update(self._weight_state())
        return state

    def _restore_state(self, state):
        for key, val in state.items():
            setattr(self, key, val)
        self._invalidate_atoms()

    # =================================================================
    # ELBO
    # =================================================================
    def elbo(self, idx=None):
        """On a subsample the local term is rescaled by N/|idx|; the global
        KL terms are never rescaled.  Returns (elbo, E[K_+])."""
        if idx is None:
            Yb, scale = self.Y, 1.0
        else:
            Yb, scale = self.Y[idx], self.N / len(idx)
        log_rho = self._log_rho(Yb)
        log_norm = logsumexp(log_rho, axis=1)
        local = scale * np.sum(log_norm)          # optimal r collapses to lse
        elbo = local - self._kl_atoms().sum() - self._kl_weights()
        r = np.exp(log_rho - log_norm[:, None])
        log_prob_empty = np.sum(np.log(np.maximum(1.0 - r, 1e-300)), axis=0)
        return float(elbo), float(np.sum(1.0 - np.exp(log_prob_empty)))

    # =================================================================
    # SVI loop
    # =================================================================
    def _svi_iteration(self, idx, rho):
        Yb = self.Y[idx]
        scale = self.N / idx.size
        r = self._responsibilities(self._log_rho(Yb))
        self._blend_atoms(rho, Yb, r, scale)
        self._blend_weights(rho, scale * r.sum(axis=0))

    def _fit_single(self, n_iter, batch_size, tau0, kappa0, rng,
                    elbo_every, elbo_idx, verbose,
                    callback=None, callback_every=0, batch_order=None):
        B = min(int(batch_size), self.N)
        for it in range(1, n_iter + 1):
            rho = (tau0 + it) ** (-kappa0)
            idx = (rng.choice(self.N, size=B, replace=False)
                   if batch_order is None else np.asarray(batch_order[it - 1]))
            self._svi_iteration(idx, rho)
            if callback is not None and callback_every and it % callback_every == 0:
                callback(self, it, time.perf_counter() - self._t0)
            if elbo_every and it % elbo_every == 0:
                L, E_K_plus = self.elbo(elbo_idx)
                self.elbo_history.append((it, L, time.perf_counter() - self._t0))
                if verbose:
                    print(f"[{self._tag}] iter {it}: rho={rho:.4f}, "
                          f"ELBO={L:.4f}, E[K_+]={E_K_plus:.3f}")

    def fit(self, n_iter=1000, batch_size=256, tau0=1.0, kappa0=0.7,
            elbo_every=0, elbo_size=None, n_restarts=1, verbose=False,
            callback=None, callback_every=0, batch_order=None):
        self._t0 = time.perf_counter()
        best_elbo, best_state = -np.inf, None
        for restart in range(n_restarts):
            seed = self.random_state + restart * 1000
            rng = np.random.default_rng(seed)
            if restart > 0:
                self._full_initialize(seed)
            elbo_idx = (None if elbo_size is None or elbo_size >= self.N
                        else rng.choice(self.N, size=elbo_size, replace=False))
            self._fit_single(n_iter, batch_size, tau0, kappa0, rng,
                             elbo_every, elbo_idx, verbose,
                             callback, callback_every, batch_order)
            L, E_K_plus = self.elbo(elbo_idx)
            self.elbo_history.append((n_iter, L, time.perf_counter() - self._t0))
            if verbose:
                print(f"  [{self._tag}] restart {restart + 1}/{n_restarts}: "
                      f"ELBO={L:.4f}, E[K_+]={E_K_plus:.3f}")
            if L > best_elbo:
                best_elbo, best_state = L, self._save_state()
        self._restore_state(best_state)
        self._compute_full_responsibilities()
        return self

    # =================================================================
    # Inference helpers
    # =================================================================
    def infer_K(self, threshold_frac=0.02, min_count=2):
        if self.r is None:
            self._compute_full_responsibilities()
        Nk = self.r.sum(axis=0)
        return int(np.sum(Nk >= max(min_count, threshold_frac * self.N))), Nk

    def _kplus_summaries(self, n_samples, seed):
        if self.r is None:
            self._compute_full_responsibilities()
        rng = np.random.default_rng(seed)
        log_prob_empty = np.sum(np.log(np.maximum(1.0 - self.r, 1e-300)), axis=0)
        prob_occupied = 1.0 - np.exp(log_prob_empty)
        draws = sample_kplus(self.r, n_samples, rng)
        pmf = np.bincount(draws, minlength=self.T + 1)[1:self.T + 1] / max(n_samples, 1)
        return dict(E_K_plus=float(np.sum(prob_occupied)), K_plus_pmf=pmf,
                    K_plus_mode=int(np.argmax(pmf)) + 1,
                    prob_occupied=prob_occupied,
                    assignments=np.argmax(self.r, axis=1))

    def predictive_log_lik(self, Y_new):
        """Held-out predictive log-likelihood per observation under q."""
        return self._predict_log_lik(Y_new, self.mixture_weights())

    def assign(self, Y_new):
        """Hard assignments of new data under the current globals."""
        return np.argmax(self._log_rho(np.asarray(Y_new, float)), axis=1)


# ========================================================
# Full-batch CAVI core
# ========================================================
class _MixtureCAVICore(_MixtureSVICore):
    """Full-batch CAVI core for baseline mixtures.

    This class is agnostic to both the atom family and the weight model.
    It reuses the same atom-family and weight-model methods as the SVI core.

    Atom family must supply
    -----------------------
        _setup_atoms
        _atom_init
        _expected_log_lik
        _blend_atoms
        _atom_stats
        _kl_atoms
        _predict_log_lik
        _atom_params
        _invalidate_atoms

    Weight model must supply
    ------------------------
        __init__
        _init_weights_prior
        _set_weights_from_counts
        _E_log_weights
        _blend_weights
        _kl_weights
        _weight_state
        mixture_weights
        posterior_summaries

    CAVI update
    -----------
    One iteration computes full-data responsibilities

        r_nk ∝ exp(E_q[log weight_k] + E_q[log f(y_n | theta_k)])

    and then performs the full-data global updates

        q(theta_k)  using r
        q(weights)  using N_k = sum_n r_nk

    The existing SVI-style natural-parameter blending routines are reused:

        _blend_atoms(damping, self.Y, r, scale=1.0)
        _blend_weights(damping, r.sum(axis=0))

    Hence damping=1 gives the ordinary CAVI coordinate update, while
    0 < damping < 1 gives a relaxed/damped CAVI update.
    """

    _tag = "CAVI"

    # ------------------------------------------------------------------
    # One full-batch CAVI iteration
    # ------------------------------------------------------------------
    def _cavi_iteration(self, damping=1.0):
        """Run one full-batch CAVI iteration.

        Parameters
        ----------
        damping : float
            Relaxation parameter.  damping=1.0 is ordinary CAVI.
            Values in (0, 1) damp the natural-parameter updates and can help
            stabilize difficult runs.
        """
        damping = float(damping)

        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must satisfy 0 < damping <= 1.")

        # Full-data local step under the current globals.
        r = self._responsibilities(self._log_rho(self.Y))

        # Full-data atom update.
        # scale=1.0 because r already covers the whole dataset.
        self._blend_atoms(damping, self.Y, r, scale=1.0)

        # Full-data weight update.
        # For finite mixtures:
        #     alpha_hat = alpha0/T + Nk
        # For DP mixtures:
        #     gamma1_hat_j = 1 + Nk_j
        #     gamma2_hat_j = E[alpha] + sum_{l>j} Nk_l
        # followed by q(alpha) update.
        self._blend_weights(damping, r.sum(axis=0))

        # The stored responsibilities are no longer guaranteed to correspond
        # exactly to the updated globals, so discard them.  They are recomputed
        # at the end of fit.
        self.r = None

    # ------------------------------------------------------------------
    # ELBO alias, for API consistency with MFM code
    # ------------------------------------------------------------------
    def compute_elbo(self, idx=None):
        return self.elbo(idx)

    # ------------------------------------------------------------------
    # CAVI fit loop
    # ------------------------------------------------------------------
    def fit(
        self,
        n_iter=500,
        batch_size=None,
        tau0=None,
        kappa0=None,
        *,
        max_iter=None,
        tol=1e-6,
        damping=1.0,
        elbo_every=1,
        elbo_size=None,
        n_restarts=1,
        min_iter=2,
        verbose=False,
        callback=None,
        callback_every=0,
        batch_order=None,
    ):
        """Fit by full-batch CAVI.

        Parameters
        ----------
        n_iter : int, default 500
            Maximum number of CAVI iterations per restart.  Kept for API
            compatibility with the SVI fit method.

        batch_size, tau0, kappa0, batch_order :
            Accepted for SVI-API compatibility but ignored by CAVI, because
            CAVI is full-batch.

        max_iter : int or None, default None
            Alias for n_iter.  If supplied, it overrides n_iter.

        tol : float or None, default 1e-6
            Stop when the absolute monitored ELBO change is below tol.
            Set tol=None to disable convergence stopping.

        damping : float, default 1.0
            Relaxation parameter.  damping=1.0 gives ordinary CAVI.
            Values in (0, 1) give damped CAVI.

        elbo_every : int, default 1
            Store and optionally print ELBO every elbo_every iterations.
            If tol is not None, the ELBO is computed every iteration for
            convergence checking, but only stored according to elbo_every
            plus the final iteration.

        elbo_size : int or None, default None
            If None, compute the full-data ELBO.  If an integer smaller than N,
            use a fixed subsample and rescale the local term by N / elbo_size.
            For formal CAVI convergence monitoring, prefer elbo_size=None.

        n_restarts : int, default 1
            Number of random restarts.  The state with the largest final
            monitored ELBO is retained.

        min_iter : int, default 2
            Minimum number of iterations before convergence stopping is allowed.

        verbose : bool, default False
            Print progress.

        callback : callable or None
            If provided, called as callback(self, it, elapsed_seconds).

        callback_every : int, default 0
            Callback frequency.  0 disables callbacks.

        Returns
        -------
        self
        """
        if max_iter is not None:
            n_iter = int(max_iter)
        else:
            n_iter = int(n_iter)

        if n_iter < 1:
            raise ValueError("n_iter/max_iter must be at least 1.")

        if verbose:
            ignored = []
            if batch_size is not None:
                ignored.append("batch_size")
            if tau0 is not None:
                ignored.append("tau0")
            if kappa0 is not None:
                ignored.append("kappa0")
            if batch_order is not None:
                ignored.append("batch_order")
            if ignored:
                print(
                    f"[{self._tag}] ignoring SVI-only argument(s): "
                    + ", ".join(ignored)
                )

        self._t0 = time.perf_counter()

        best_elbo = -np.inf
        best_state = None

        for restart in range(n_restarts):
            seed = self.random_state + restart * 1000
            rng = np.random.default_rng(seed)

            if restart > 0:
                self._full_initialize(seed)

            elbo_idx = (
                None
                if elbo_size is None or elbo_size >= self.N
                else rng.choice(self.N, size=elbo_size, replace=False)
            )

            prev_L = None
            final_L = -np.inf
            final_E_K_plus = float("nan")
            last_it = 0

            for it in range(1, n_iter + 1):
                last_it = it

                self._cavi_iteration(damping=damping)

                if (
                    callback is not None
                    and callback_every
                    and it % callback_every == 0
                ):
                    callback(self, it, time.perf_counter() - self._t0)

                # If tol is enabled, compute ELBO every iteration for stopping.
                need_elbo = (
                    tol is not None
                    or (elbo_every and it % elbo_every == 0)
                    or it == n_iter
                )

                if need_elbo:
                    L, E_K_plus = self.elbo(elbo_idx)
                    final_L = L
                    final_E_K_plus = E_K_plus

                    should_store = (
                        it == n_iter
                        or (elbo_every and it % elbo_every == 0)
                    )

                    if should_store:
                        self.elbo_history.append(
                            (it, L, time.perf_counter() - self._t0)
                        )

                    if verbose and should_store:
                        print(
                            f"[{self._tag}] restart {restart + 1}/{n_restarts}, "
                            f"iter {it}: "
                            f"ELBO={L:.6f}, "
                            f"E[K_+]={E_K_plus:.3f}"
                        )

                    if (
                        tol is not None
                        and prev_L is not None
                        and it >= min_iter
                    ):
                        delta = abs(L - prev_L)

                        if delta < tol * max(1.0, abs(prev_L)):
                            if verbose:
                                print(
                                    f"[{self._tag}] restart {restart + 1}/{n_restarts} "
                                    f"converged at iter {it}; "
                                    f"|ΔELBO|={delta:.3e}"
                                )

                            # Ensure the final converged point is in history
                            # even if elbo_every did not request storage.
                            if (
                                not self.elbo_history
                                or self.elbo_history[-1][0] != it
                            ):
                                self.elbo_history.append(
                                    (it, L, time.perf_counter() - self._t0)
                                )

                            break

                    prev_L = L

            # If no ELBO was computed inside the loop, compute it now.
            if not np.isfinite(final_L):
                final_L, final_E_K_plus = self.elbo(elbo_idx)

            # Ensure final state of this restart is recorded.
            if (
                not self.elbo_history
                or self.elbo_history[-1][0] != last_it
            ):
                self.elbo_history.append(
                    (last_it, final_L, time.perf_counter() - self._t0)
                )

            if verbose:
                print(
                    f"  [{self._tag}] restart {restart + 1}/{n_restarts}: "
                    f"ELBO={final_L:.6f}, E[K_+]={final_E_K_plus:.3f}"
                )

            if final_L > best_elbo:
                best_elbo = final_L
                best_state = self._save_state()

        self._restore_state(best_state)
        self._compute_full_responsibilities()

        return self

# ========================================================
# DP mixture: truncated stick breaking, alpha ~ Gamma(s1, s2)
# ========================================================
class DPWeights:
    _tag = "DP-SVI"

    def __init__(self, Y, T, s1=1.0, s2=1.0, init="kmeans", init_size=None,
                 random_state=42, init_data=None, init_scale=None, **atom_kw):
        self.s1, self.s2 = float(s1), float(s2)
        self._setup(Y, T, init, init_size, random_state,
                    init_data=init_data, init_scale=init_scale, **atom_kw)
        self._full_initialize(self.random_state)

    def _init_weights_prior(self):
        self.gamma1_hat = np.ones(self.T - 1)
        self.gamma2_hat = np.ones(self.T - 1)
        self.s1_hat, self.s2_hat = self.s1, self.s2

    def _stick_params(self, Nk):
        tail = np.cumsum(Nk[::-1])[::-1]
        return 1.0 + Nk[:-1], self.s1_hat / self.s2_hat + tail[1:]

    def _set_weights_from_counts(self, Nk):
        self.gamma1_hat, self.gamma2_hat = self._stick_params(Nk)
        self._update_q_alpha()

    def _E_log_stick_weights(self):
        g12 = self.gamma1_hat + self.gamma2_hat
        E_log_V = psi(self.gamma1_hat) - psi(g12)
        E_log_1mV = psi(self.gamma2_hat) - psi(g12)
        cum = np.concatenate(([0.0], np.cumsum(E_log_1mV)))
        E_log_pi = np.concatenate((E_log_V + cum[:-1], [cum[-1]]))
        return E_log_pi, E_log_V, E_log_1mV

    def _E_log_weights(self):
        return self._E_log_stick_weights()[0]

    def _update_q_alpha(self):
        _, _, E_log_1mV = self._E_log_stick_weights()
        self.s1_hat = self.s1 + (self.T - 1)
        self.s2_hat = max(1e-6, self.s2 - np.sum(E_log_1mV))

    def _blend_weights(self, rho, N_tilde):
        g1_c, g2_c = self._stick_params(N_tilde)
        self.gamma1_hat = (1.0 - rho) * self.gamma1_hat + rho * g1_c
        self.gamma2_hat = (1.0 - rho) * self.gamma2_hat + rho * g2_c
        self._update_q_alpha()

    def _kl_weights(self):
        g1, g2 = self.gamma1_hat, self.gamma2_hat
        _, E_log_V, E_log_1mV = self._E_log_stick_weights()
        E_alpha = self.s1_hat / self.s2_hat
        E_log_alpha = psi(self.s1_hat) - np.log(self.s2_hat)
        log_B = gammaln(g1) + gammaln(g2) - gammaln(g1 + g2)
        E_log_q = -log_B + (g1 - 1.0) * E_log_V + (g2 - 1.0) * E_log_1mV
        E_log_p = E_log_alpha + (E_alpha - 1.0) * E_log_1mV
        kl_V = np.sum(E_log_q - E_log_p)
        a1, b1, a0, b0 = self.s1_hat, self.s2_hat, self.s1, self.s2
        kl_alpha = ((a1 - a0) * psi(a1) - gammaln(a1) + gammaln(a0)
                    + a0 * (np.log(b1) - np.log(b0)) + a1 * (b0 - b1) / b1)
        return kl_V + kl_alpha

    def _weight_state(self):
        return dict(gamma1_hat=self.gamma1_hat.copy(),
                    gamma2_hat=self.gamma2_hat.copy(),
                    s1_hat=self.s1_hat, s2_hat=self.s2_hat)

    def mixture_weights(self):
        g12 = self.gamma1_hat + self.gamma2_hat
        E_V, E_1mV = self.gamma1_hat / g12, self.gamma2_hat / g12
        cum = np.concatenate(([1.0], np.cumprod(E_1mV)))
        return np.concatenate((E_V * cum[:-1], [cum[-1]]))

    def posterior_summaries(self, n_samples=2000, seed=None):
        out = self._kplus_summaries(n_samples, seed)
        out["E_pi"] = self.mixture_weights()
        out["E_alpha"] = float(self.s1_hat / self.s2_hat)
        return out


# ========================================================
# Fixed-T finite mixture, eta ~ Dir(alpha0/T, ..., alpha0/T)
# ========================================================
class FiniteWeights:
    """alpha0 = T gives Dir(1, ..., 1).  The overfitted-mixture regime
    (Rousseau & Mengersen) is alpha0 / T < d_theta / 2."""
    _tag = "Finite-T SVI"

    def __init__(self, Y, T, alpha0=1.0, init="kmeans", init_size=None,
                 random_state=42, init_data=None, init_scale=None, **atom_kw):
        self.alpha0 = float(alpha0)
        self._setup(Y, T, init, init_size, random_state,
                    init_data=init_data, init_scale=init_scale, **atom_kw)
        self._full_initialize(self.random_state)

    def _init_weights_prior(self):
        self.alpha_hat = np.full(self.T, self.alpha0 / self.T)

    def _set_weights_from_counts(self, Nk):
        self.alpha_hat = self.alpha0 / self.T + Nk

    def _E_log_weights(self):
        return psi(self.alpha_hat) - psi(np.sum(self.alpha_hat))

    def _blend_weights(self, rho, N_tilde):
        self.alpha_hat = ((1.0 - rho) * self.alpha_hat
                          + rho * (self.alpha0 / self.T + N_tilde))

    def _kl_weights(self):
        a_hat = self.alpha_hat
        a0 = np.full(self.T, self.alpha0 / self.T)
        return (gammaln(a_hat.sum()) - gammaln(a_hat).sum()
                - gammaln(a0.sum()) + gammaln(a0).sum()
                + np.sum((a_hat - a0) * (psi(a_hat) - psi(a_hat.sum()))))

    def _weight_state(self):
        return dict(alpha_hat=self.alpha_hat.copy())

    def mixture_weights(self):
        return self.alpha_hat / self.alpha_hat.sum()

    def posterior_summaries(self, n_samples=2000, seed=None):
        out = self._kplus_summaries(n_samples, seed)
        out["E_eta"] = self.mixture_weights()
        return out


# ========================================================
# Composed models: atom family x weight model
# ========================================================
def compose(name, atom_cls, weight_cls, tag=None):
    cls = type(name, (atom_cls, weight_cls, _MixtureSVICore), {})
    if tag is not None:
        cls._tag = tag
    return cls

def compose_cavi(name, atom_cls, weight_cls, tag=None):
    cls = type(name, (atom_cls, weight_cls, _MixtureCAVICore), {})
    if tag is not None:
        cls._tag = tag
    return cls

DPMixtureSVI_BetaBernoulli = compose("DPMixtureSVI_BetaBernoulli", BetaBernoulliAtoms,
                        DPWeights, "DP-SVI")
FiniteMixtureSVI_BetaBernoulli = compose("FiniteMixtureSVI_BetaBernoulli", BetaBernoulliAtoms,
                            FiniteWeights, "Finite-T SVI")
DPMixtureSVI_Wishart = compose("DPMixtureSVI_Wishart", ZeroMeanWishartAtoms,
                        DPWeights, "DP-SVI")
FiniteMixtureSVI_Wishart = compose("FiniteMixtureSVI_Wishart", ZeroMeanWishartAtoms,
                            FiniteWeights, "Finite-T SVI")
DPMixtureSVI_Gaussian = compose("DPMixtureSVI_Gaussian", NormalWishartAtoms,
                                DPWeights, "DP-SVI (Gaussian)")
FiniteMixtureSVI_Gaussian = compose("FiniteMixtureSVI_Gaussian",
                                    NormalWishartAtoms, FiniteWeights,
                                    "Finite-T SVI (Gaussian)")
DPMixtureSVI_StudentT = compose("DPMixtureSVI_StudentT", StudentTAtoms,
                                DPWeights, "DP-SVI (t)")
FiniteMixtureSVI_StudentT = compose("FiniteMixtureSVI_StudentT",
                                    StudentTAtoms, FiniteWeights,
                                    "Finite-T SVI (t)")

# ========================================================
# Composed CAVI models: atom family x weight model
# ========================================================

DPMixtureCAVI_BetaBernoulli = compose_cavi(
    "DPMixtureCAVI_BetaBernoulli",
    BetaBernoulliAtoms,
    DPWeights,
    "DP-CAVI",
)

FiniteMixtureCAVI_BetaBernoulli = compose_cavi(
    "FiniteMixtureCAVI_BetaBernoulli",
    BetaBernoulliAtoms,
    FiniteWeights,
    "Finite-T CAVI",
)

DPMixtureCAVI_Wishart = compose_cavi(
    "DPMixtureCAVI_Wishart",
    ZeroMeanWishartAtoms,
    DPWeights,
    "DP-CAVI",
)

FiniteMixtureCAVI_Wishart = compose_cavi(
    "FiniteMixtureCAVI_Wishart",
    ZeroMeanWishartAtoms,
    FiniteWeights,
    "Finite-T CAVI",
)

DPMixtureCAVI_Gaussian = compose_cavi(
    "DPMixtureCAVI_Gaussian",
    NormalWishartAtoms,
    DPWeights,
    "DP-CAVI (Gaussian)",
)

FiniteMixtureCAVI_Gaussian = compose_cavi(
    "FiniteMixtureCAVI_Gaussian",
    NormalWishartAtoms,
    FiniteWeights,
    "Finite-T CAVI (Gaussian)",
)

DPMixtureCAVI_StudentT = compose_cavi(
    "DPMixtureCAVI_StudentT",
    StudentTAtoms,
    DPWeights,
    "DP-CAVI (t)",
)

FiniteMixtureCAVI_StudentT = compose_cavi(
    "FiniteMixtureCAVI_StudentT",
    StudentTAtoms,
    FiniteWeights,
    "Finite-T CAVI (t)",
)

__all__ = [
    "_MixtureSVICore",
    "_MixtureCAVICore",
    "DPWeights",
    "FiniteWeights",
    "compose",
    "compose_cavi",

    # SVI classes
    "DPMixtureSVI_BetaBernoulli",
    "FiniteMixtureSVI_BetaBernoulli",
    "DPMixtureSVI_Wishart",
    "FiniteMixtureSVI_Wishart",
    "DPMixtureSVI_Gaussian",
    "FiniteMixtureSVI_Gaussian",
    "DPMixtureSVI_StudentT",
    "FiniteMixtureSVI_StudentT",

    # CAVI classes
    "DPMixtureCAVI_BetaBernoulli",
    "FiniteMixtureCAVI_BetaBernoulli",
    "DPMixtureCAVI_Wishart",
    "FiniteMixtureCAVI_Wishart",
    "DPMixtureCAVI_Gaussian",
    "FiniteMixtureCAVI_Gaussian",
    "DPMixtureCAVI_StudentT",
    "FiniteMixtureCAVI_StudentT",
]