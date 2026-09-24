"""Dynamic mixture of finite mixtures (MFM), UNTIED variant: stochastic VI
with a separate Dirichlet q(eta_kappa) for every candidate kappa.

This is the reference implementation that tied_mfm.py approximates.  Keep it
for comparison; use tied_mfm.py at large T.

Model
-----
    K ~ p_K (BetaNegBinomial, truncated at T)      alpha ~ Gamma(a0, b0)
    eta_K | K, alpha ~ Dir(alpha/K, ..., alpha/K)
    z_i | eta_K ~ Cat(eta_K)       theta_k ~ G0       y_i | z_i ~ f(y | theta_{z_i})

Variational family
------------------
    q(K) = Cat(pi)      q(alpha) = Gamma(a_eta, b_eta)
    q(theta_k), k = 1..T      shared atoms, one family (see component_fams/)
    q(eta_kappa) = Dir(alpha_hat_kappa)     ONE Dirichlet PER CANDIDATE
    q(z_i | K = kappa) = Cat(r_i^kappa)

Untied vs tied
--------------
Here each candidate owns its own alpha_hat_kappa (length kappa), so the local
step loops over active candidates: O(B T^2) per iteration, and a (B, kappa)
responsibility matrix per candidate if retained.  tied_mfm.py replaces the
T Dirichlets by one Dir(a_1..a_T) whose renormalised prefixes are the
candidates' Dirichlets, which makes the local step O(B T) with no loop.  The
two agree to displayed precision on MAP K, E[K_+] and E[alpha] at T <= 40
(see README_longtail.md); the tied version is ~50x faster at T = 200.

Two places where the untied version is the *exact* one and the tied version
approximates: (1) the coordinate update alpha_hat_kappa = alpha_bar/kappa +
counts is closed-form here, while the tied a_k has no closed form; (2) atom
sorting is exact under tying but needs a heuristic here (an atom moving into
candidate kappa's window restarts from alpha_bar/kappa, see _reorder_atoms).

Memory
------
The local step is streamed over candidates: r^kappa is formed one candidate
at a time and only the aggregates (ell, counts, w, E[K_+]) are kept.
Materialising every r^kappa costs sum_kappa N kappa 8 bytes (~66 GB at
N = 7291, T = 150).  Full responsibilities are retained at the end only for
candidates carrying q(K) mass >= keep_r_tol.

Known limitation in non-stationary streams
------------------------------------------
Atom updates are gated by w_ik = sum_{kappa>=k} pi_kappa r^kappa_ik.  Once
q(K) concentrates at kappa*, atoms beyond kappa* receive no data and cannot
be opened by data that arrives later.  Dense initialisation avoids this;
sparse init in a population-ordered stream triggers it.  Shared with the
tied version; a principled exploration mechanism is future work.

Initialisation and streaming
----------------------------
Atoms are initialised from `init_data` if given (else a subsample of Y of
size `init_size`) with pseudo-counts scaled by `init_scale` (default
N / n_init, commensurate with the N/B-scaled minibatch updates -- an unscaled
init is overwritten by the first few batches).  `atom_kw` such as
`B0_mode="cov", prior_data=calibration_buffer` reach the atom family.
fit(..., batch_order=[idx_1, idx_2, ...]) processes batches in that order.
Note: trust_region.py and stream_replay.py operate on the tied parameters
(a, log_pi) and do not support this class.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import time
import numpy as np
from scipy.special import psi, gammaln, logsumexp, xlogy
from scipy.stats import betanbinom

from CAVI.component_fams.betabernoulli_atoms import BetaBernoulliAtoms, sample_kplus
from CAVI.component_fams.gaussian_atoms import NormalWishartAtoms
from CAVI.component_fams.studentt_atoms import StudentTAtoms
from CAVI.component_fams.wishart_atoms import ZeroMeanWishartAtoms


class _MFMCore:
    """Atom-family-agnostic untied dynamic MFM.  Compose with an atom family."""
    
    def __init__(self, Y, T, a0=1.0, b0=0.3,
                 alpha_lambda_prior=1.0, alpha_pi_prior=4.0, beta_pi_prior=3.0,
                 init="kmeans", init_size=None, init_data=None, init_scale=None,
                 keep_r_tol=1e-4, sort_atoms=False, random_state=42, **atom_kw):
        self.Y = np.asarray(Y, dtype=float)
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        self.N, self.p = self.Y.shape
        self.T = int(T)
        self.kappas = np.arange(1, self.T + 1)
        self.init, self.init_size = init, init_size
        self.init_data = None if init_data is None else np.asarray(init_data, float)
        self.init_scale = init_scale
        self.keep_r_tol = float(keep_r_tol)
        self.sort_atoms = bool(sort_atoms)
        self.random_state = random_state
        self.a0, self.b0 = float(a0), float(b0)
        self._setup_atoms(**atom_kw)

        log_pK = betanbinom.logpmf(np.arange(self.T), alpha_lambda_prior,
                                   alpha_pi_prior, beta_pi_prior)
        self.log_pT_K = log_pK - logsumexp(log_pK)
        self._full_initialize(self.random_state)

    # =================================================================
    # Local step -- looped over candidates (the untied cost)
    # =================================================================
    def _E_log_eta(self, kappa):
        a = self.alpha_hat[kappa - 1]
        return psi(a) - psi(a.sum())

    def _local_step(self, E_log_lik, keep_r=False, accumulate_w=False,
                    want_kplus=False):
        """One pass over the active candidates.  Returns a dict with
            ell[kappa-1]    : (n,) logsumexp_k log rho_ik^kappa
            counts[kappa-1] : (kappa,) sum_i r_ik^kappa
            w               : (n, T) sum_kappa pi_kappa r^kappa, or None
            E_K_plus        : sum_kappa pi_kappa sum_k (1 - prod_i (1 - r_ik))
            r[kappa-1]      : (n, kappa) where keep_r says so, else None
        `keep_r` may be a bool or a boolean mask over candidates."""
        n = E_log_lik.shape[0]
        keep_mask = (np.full(self.T, bool(keep_r)) if np.isscalar(keep_r)
                     else np.asarray(keep_r, dtype=bool))
        r, ell, counts = [None] * self.T, [None] * self.T, [None] * self.T
        w = np.zeros((n, self.T)) if accumulate_w else None
        kplus = 0.0
        for kappa in self.kappas[self.active]:
            log_rho = self._E_log_eta(kappa)[None, :] + E_log_lik[:, :kappa]
            lse = logsumexp(log_rho, axis=1)
            rk = np.exp(log_rho - lse[:, None])
            ell[kappa - 1] = lse
            counts[kappa - 1] = rk.sum(axis=0)
            if accumulate_w:
                w[:, :kappa] += self.pi[kappa - 1] * rk
            if want_kplus:
                log_empty = np.sum(np.log(np.maximum(1.0 - rk, 1e-300)), axis=0)
                kplus += self.pi[kappa - 1] * np.sum(1.0 - np.exp(log_empty))
            if keep_mask[kappa - 1]:
                r[kappa - 1] = rk
        return dict(r=r, ell=ell, counts=counts, w=w, E_K_plus=float(kplus))

    def _resp_for(self, E_log_lik, kappa):
        log_rho = self._E_log_eta(kappa)[None, :] + E_log_lik[:, :kappa]
        return np.exp(log_rho - logsumexp(log_rho, axis=1, keepdims=True))

    def _keep_mask(self):
        mask = self.active & (self.pi >= self.keep_r_tol)
        mask[np.argmax(self.pi)] = True
        return mask

    # =================================================================
    # Candidate score
    # =================================================================
    @staticmethod
    def _dirichlet_entropy(a):
        a0 = a.sum()
        return (gammaln(a).sum() - gammaln(a0) + (a0 - a.size) * psi(a0)
                - np.sum((a - 1.0) * psi(a)))

    def _F_tilde_global(self, kappa):
        """E[log p(eta_kappa | kappa, alpha)] + H[q(eta_kappa)], log-Gamma
        terms at alpha_bar = E[alpha] (plug-in)."""
        a_hat = self.alpha_hat[kappa - 1]
        E_log_eta = psi(a_hat) - psi(a_hat.sum())
        ab = self.alpha_bar
        E_log_alpha = psi(self.a_eta_hat) - np.log(self.b_eta_hat)
        return ((kappa - 1) * E_log_alpha - kappa * np.log(kappa)
                + gammaln(ab + 1.0) - kappa * gammaln(ab / kappa + 1.0)
                + (ab / kappa - 1.0) * E_log_eta.sum()
                + self._dirichlet_entropy(a_hat))

    # =================================================================
    # Initialisation / state
    # =================================================================
    def _init_atoms(self, random_state):
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
        self.active = np.ones(self.T, dtype=bool)
        self.a_eta_hat, self.b_eta_hat = self.a0, self.b0
        self.alpha_bar = self.a_eta_hat / self.b_eta_hat
        Y0, scale = self._init_atoms(random_state)

        # local bound under uniform weights eta_kappa = 1/kappa (streamed)
        E = self._expected_log_lik(Y0)
        log_evidence = np.empty(self.T)
        self.alpha_hat = [None] * self.T
        for kappa in self.kappas:
            log_rho = E[:, :kappa] - np.log(kappa)
            lse = logsumexp(log_rho, axis=1)
            rk = np.exp(log_rho - lse[:, None])
            log_evidence[kappa - 1] = scale * lse.sum()
            self.alpha_hat[kappa - 1] = self.alpha_bar / kappa + scale * rk.sum(axis=0)
        log_pi = self.log_pT_K + log_evidence
        self.log_pi = log_pi - logsumexp(log_pi)
        self.pi = np.exp(self.log_pi)
        self._update_q_alpha_eta()

        self.elbo_history, self.active_history = [], []
        self.r, self._E_K_plus = None, float("nan")

    def _save_state(self):
        st = dict(pi=self.pi.copy(), log_pi=self.log_pi.copy(),
                  active=self.active.copy(),
                  a_eta_hat=self.a_eta_hat, b_eta_hat=self.b_eta_hat,
                  alpha_bar=self.alpha_bar,
                  alpha_hat=[None if a is None else a.copy() for a in self.alpha_hat],
                  elbo_history=list(self.elbo_history),
                  active_history=list(self.active_history))
        st.update({k: np.copy(getattr(self, k)) for k in self._atom_params})
        return st

    def _restore_state(self, st):
        for k, v in st.items():
            setattr(self, k, v)
        self._invalidate_atoms()

    # =================================================================
    # SVI iteration
    # =================================================================
    def _svi_iteration(self, idx, rho):
        Yb = self.Y[idx]
        scale = self.N / idx.size
        act = self.active

        out = self._local_step(self._expected_log_lik(Yb), accumulate_w=True)
        ell, counts, w = out["ell"], out["counts"], out["w"]

        phi = np.full(self.T, -np.inf)
        for kappa in self.kappas[act]:
            phi[kappa - 1] = (self.log_pT_K[kappa - 1] + self._F_tilde_global(kappa)
                              + scale * ell[kappa - 1].sum())

        self._blend_atoms(rho, Yb, w, scale)

        for kappa in self.kappas[act]:                     # exact coordinate step
            a_check = self.alpha_bar / kappa + scale * counts[kappa - 1]
            self.alpha_hat[kappa - 1] = (1 - rho) * self.alpha_hat[kappa - 1] + rho * a_check

        new = np.full(self.T, -np.inf)
        new[act] = (1 - rho) * self.log_pi[act] + rho * phi[act]
        self.log_pi = new - logsumexp(new)
        self.pi = np.exp(self.log_pi)

        if self.sort_atoms:
            self._reorder_atoms()
        self._update_q_alpha_eta()

    def mixture_weights(self):
        """Marginal weight of atom k: sum_{kappa>=k} pi_kappa E[eta_{kappa,k}]."""
        w = np.zeros(self.T)
        for kappa in self.kappas[self.active]:
            a = self.alpha_hat[kappa - 1]
            w[:kappa] += self.pi[kappa - 1] * (a / a.sum())
        return w

    def _reorder_atoms(self):
        """Sort atoms by marginal weight (removes the shared-atom ordering
        artefact that pins q(K) to the truncation).  Heuristic here: an
        atom that moves into candidate kappa's window restarts from the
        prior alpha_bar / kappa.  Exact under the tied family."""
        order = np.argsort(-self.mixture_weights(), kind="stable")
        if np.array_equal(order, np.arange(self.T)):
            return
        for name in self._atom_params:
            setattr(self, name, getattr(self, name)[order])
        self._invalidate_atoms()
        for kappa in self.kappas[self.active]:
            full = np.zeros(self.T)
            full[:kappa] = self.alpha_hat[kappa - 1]       # all > 0
            new = full[order][:kappa]
            new[new <= 0.0] = self.alpha_bar / kappa       # newly visible atoms
            self.alpha_hat[kappa - 1] = new

    def _update_q_alpha_eta(self):
        ab = self.a_eta_hat / self.b_eta_hat
        self.a_eta_hat = self.a0 + np.sum(self.pi * self.kappas) - 1.0
        s = 0.0
        for kappa in self.kappas[self.active]:
            s += self.pi[kappa - 1] * (psi(ab + 1.0) - psi(ab / kappa + 1.0)
                                       + self._E_log_eta(kappa).sum() / kappa)
        self.b_eta_hat = max(1e-6, self.b0 - s)
        self.alpha_bar = self.a_eta_hat / self.b_eta_hat

    # =================================================================
    # ELBO
    # =================================================================
    def _gamma_kl(self):
        a1, b1, a0, b0 = self.a_eta_hat, self.b_eta_hat, self.a0, self.b0
        return ((a1 - a0) * psi(a1) - gammaln(a1) + gammaln(a0)
                + a0 * (np.log(b1) - np.log(b0)) + a1 * (b0 - b1) / b1)

    def compute_elbo(self, idx=None):
        """(elbo, E[K_+]).  On a subsample the local term is rescaled by
        N / |idx|; global terms are not."""
        Yb, scale = (self.Y, 1.0) if idx is None else (self.Y[idx], self.N / len(idx))
        act = self.active
        out = self._local_step(self._expected_log_lik(Yb), want_kplus=True)
        F = np.zeros(self.T)
        for kappa in self.kappas[act]:
            F[kappa - 1] = self._F_tilde_global(kappa) + scale * out["ell"][kappa - 1].sum()
        elbo = np.sum(self.pi[act] * (self.log_pT_K[act] + F[act]))
        elbo -= np.sum(xlogy(self.pi[act], self.pi[act]))
        elbo -= self._gamma_kl() + self._kl_atoms().sum()
        return float(elbo), out["E_K_plus"]

    # =================================================================
    # Candidate pruning (monotone; pruned candidates have pi = 0)
    # =================================================================
    def _apply_active(self, keep):
        keep = keep & self.active
        if not keep.any():
            return 0
        n = int(self.active.sum() - keep.sum())
        if n == 0:
            return 0
        self.active = keep
        self.log_pi[~keep] = -np.inf
        self.log_pi -= logsumexp(self.log_pi)
        self.pi = np.exp(self.log_pi)
        self._update_q_alpha_eta()
        return n

    def _prune_by_kplus(self, E_K_plus, margin):
        """Stage 1: drop kappa < E[K_+] - margin (K >= K_+ a.s.).  Start
        this late enough that E[K_+] has stabilised; it is irreversible."""
        keep = ~(self.kappas < E_K_plus - margin)
        keep[np.argmax(self.pi)] = True
        return self._apply_active(keep)

    def _prune_by_posterior(self, tol):
        """Stage 2: keep the contiguous range carrying q(K) mass >= 1 - tol."""
        act = np.where(self.active)[0]
        if act.size <= 1:
            return 0
        order = act[np.argsort(-self.pi[act])]
        cum = np.cumsum(self.pi[order])
        kept = order[:int(np.searchsorted(cum, 1 - tol)) + 1]
        keep = np.zeros(self.T, bool)
        keep[kept.min():kept.max() + 1] = True
        return self._apply_active(keep)

    @staticmethod
    def _due(it, after, every):
        if not after or it < after:
            return False
        return it == after or (every and (it - after) % every == 0)

    # =================================================================
    # Fit
    # =================================================================
    def fit(self, n_iter=1000, batch_size=256, tau0=1.0, kappa0=0.7,
            elbo_every=0, elbo_size=None, n_restarts=1,
            prune_after=None, prune_every=0, prune_margin=5,
            prune_pi_after=None, prune_pi_every=0, prune_pi_tol=1e-3,
            verbose=False, callback=None, callback_every=0, batch_order=None):
        """Pruning schedule is in iterations; express it in epochs from the
        caller (after ~10 epochs, margin 5 recovers the no-prune answer)."""
        self._t0 = time.perf_counter()
        best, best_state = -np.inf, None
        B = min(int(batch_size), self.N)
        for restart in range(n_restarts):
            seed = self.random_state + 1000 * restart
            rng = np.random.default_rng(seed)
            if restart > 0:
                self._full_initialize(seed)
            elbo_idx = (None if elbo_size is None or elbo_size >= self.N
                        else rng.choice(self.N, elbo_size, replace=False))
            for it in range(1, n_iter + 1):
                rho = (tau0 + it) ** (-kappa0)
                idx = (rng.choice(self.N, size=B, replace=False)
                       if batch_order is None else np.asarray(batch_order[it - 1]))
                self._svi_iteration(idx, rho)
                if self._due(it, prune_after, prune_every):
                    _, kp = self.compute_elbo(elbo_idx)
                    n = self._prune_by_kplus(kp, prune_margin)
                    if verbose and n:
                        print(f"iter {it}: K+ prune {n}, E[K+]={kp:.2f}")
                if self._due(it, prune_pi_after, prune_pi_every):
                    n = self._prune_by_posterior(prune_pi_tol)
                    if verbose and n:
                        print(f"iter {it}: q(K) prune {n}, MAP K={self.kappas[np.argmax(self.pi)]}")
                self.active_history.append((it, int(self.active.sum())))
                if callback is not None and callback_every and it % callback_every == 0:
                    callback(self, it, time.perf_counter() - self._t0)
                if elbo_every and it % elbo_every == 0:
                    L, kp = self.compute_elbo(elbo_idx)
                    self.elbo_history.append((it, L, time.perf_counter() - self._t0))
                    if verbose:
                        print(f"iter {it}: ELBO {L:.4f}  E[K+] {kp:.2f}  "
                              f"MAP K {self.kappas[np.argmax(self.pi)]}  "
                              f"E[alpha] {self.alpha_bar:.2f}  active {self.active.sum()}")
            L, kp = self.compute_elbo(elbo_idx)
            self.elbo_history.append((n_iter, L, time.perf_counter() - self._t0))
            if L > best:
                best, best_state = L, self._save_state()
        self._restore_state(best_state)
        self._finalize_responsibilities()
        return self

    def _finalize_responsibilities(self):
        out = self._local_step(self._expected_log_lik(self.Y),
                               keep_r=self._keep_mask(), want_kplus=True)
        self.r, self._E_K_plus = out["r"], out["E_K_plus"]

    # =================================================================
    # Summaries
    # =================================================================
    def predictive_log_lik(self, Y_new):
        return self._predict_log_lik(Y_new, self.mixture_weights())

    def assign(self, Y_new):
        """Hard assignments of new data at the MAP K."""
        E = self._expected_log_lik(np.asarray(Y_new, float))
        kmap = int(self.kappas[np.argmax(self.pi)])
        return np.argmax(self._resp_for(E, kmap), axis=1)

    def posterior_summaries(self, n_samples=500, seed=None):
        """q_K is the posterior over K; K_plus_pmf is the MC pmf of the
        number of occupied components, with kappa drawn from q(K).  Compare
        methods on E_K_plus / K_plus_pmf; report q_K separately."""
        if self.r is None:
            self._finalize_responsibilities()
        rng = np.random.default_rng(seed)
        have = np.array([rk is not None for rk in self.r])
        p = np.where(have, self.pi, 0.0)
        p = p / p.sum()
        draws = rng.choice(self.kappas, size=n_samples, p=p)
        Kplus = np.empty(n_samples, dtype=int)
        for kappa in np.unique(draws):
            sel = draws == kappa
            Kplus[sel] = sample_kplus(self.r[kappa - 1], int(sel.sum()), rng)
        pmf = np.bincount(Kplus, minlength=self.T + 1)[1:self.T + 1] / n_samples
        kmap = int(self.kappas[np.argmax(self.pi)])
        return dict(q_K=self.pi.copy(), active=self.active.copy(),
                    E_K=float(np.sum(self.pi * self.kappas)),
                    K_plus_pmf=pmf, E_K_plus=float(self._E_K_plus),
                    K_plus_mode=int(np.argmax(pmf)) + 1, K_map=kmap,
                    E_alpha=float(self.a_eta_hat / self.b_eta_hat),
                    mixture_weights=self.mixture_weights(),
                    assignments=np.argmax(self.r[kmap - 1], axis=1))

    _local = _local_step

class _CAVIMFMCore(_MFMCore):
    def _update_qK_from_local(self, ell, damping=1.0):
        act = self.active
        phi = np.full(self.T, -np.inf)

        for kappa in self.kappas[act]:
            phi[kappa - 1] = (
                self.log_pT_K[kappa - 1]
                + self._F_tilde_global(kappa)
                + ell[kappa - 1].sum()
            )

        target_log_pi = phi.copy()
        target_log_pi[act] -= logsumexp(target_log_pi[act])

        if damping >= 1.0:
            self.log_pi = target_log_pi
        else:
            new = np.full(self.T, -np.inf)
            new[act] = (
                (1.0 - damping) * self.log_pi[act]
                + damping * target_log_pi[act]
            )
            new[act] -= logsumexp(new[act])
            self.log_pi = new

        self.pi = np.exp(self.log_pi)

    def _cavi_iteration(self, damping=1.0):
        damping = float(damping)
        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must satisfy 0 < damping <= 1.")

        # Current full-data local pass: get counts and atom weights.
        E_old = self._expected_log_lik(self.Y)
        out_old = self._local_step(E_old, accumulate_w=True)

        counts = out_old["counts"]
        w = out_old["w"]

        # Full-data atom update.  damping=1 gives exact CAVI atom update.
        self._blend_atoms(damping, self.Y, w, scale=1.0)

        # Full-data eta updates.
        for kappa in self.kappas[self.active]:
            target = self.alpha_bar / kappa + counts[kappa - 1]

            if damping >= 1.0:
                self.alpha_hat[kappa - 1] = target
            else:
                self.alpha_hat[kappa - 1] = (
                    (1.0 - damping) * self.alpha_hat[kappa - 1]
                    + damping * target
                )

        if self.sort_atoms:
            self._reorder_atoms()

        # Fresh local pass under updated atoms/etas for q(K).
        E_new = self._expected_log_lik(self.Y)
        out_new = self._local_step(E_new)

        self._update_qK_from_local(out_new["ell"], damping=damping)

        # q(alpha_eta) update using updated q(K), q(eta_kappa).
        self._update_q_alpha_eta()

    def fit(
        self,
        max_iter=500,
        tol=1e-6,
        damping=1.0,
        elbo_every=1,
        elbo_size=None,
        n_restarts=1,
        prune_after=None,
        prune_every=0,
        prune_margin=5,
        prune_pi_after=None,
        prune_pi_every=0,
        prune_pi_tol=1e-3,
        min_iter=1,
        verbose=False,
        callback=None,
        callback_every=0,
    ):
        self._t0 = time.perf_counter()

        best = -np.inf
        best_state = None
        max_iter = int(max_iter)

        for restart in range(n_restarts):
            seed = self.random_state + 1000 * restart
            rng = np.random.default_rng(seed)

            if restart > 0:
                self._full_initialize(seed)

            elbo_idx = (
                None
                if elbo_size is None or elbo_size >= self.N
                else rng.choice(self.N, elbo_size, replace=False)
            )

            prev_L = None
            final_L = -np.inf

            for it in range(1, max_iter + 1):
                self._cavi_iteration(damping=damping)

                pruned_this_iter = False

                if self._due(it, prune_after, prune_every):
                    _, kp = self.compute_elbo(elbo_idx)
                    n_pruned = self._prune_by_kplus(kp, prune_margin)
                    pruned_this_iter = pruned_this_iter or (n_pruned > 0)

                    if verbose and n_pruned:
                        print(f"iter {it}: K+ prune {n_pruned}, E[K+]={kp:.2f}")

                if self._due(it, prune_pi_after, prune_pi_every):
                    n_pruned = self._prune_by_posterior(prune_pi_tol)
                    pruned_this_iter = pruned_this_iter or (n_pruned > 0)

                    if verbose and n_pruned:
                        print(
                            f"iter {it}: q(K) prune {n_pruned}, "
                            f"MAP K={self.kappas[np.argmax(self.pi)]}"
                        )

                self.active_history.append((it, int(self.active.sum())))

                if (
                    callback is not None
                    and callback_every
                    and it % callback_every == 0
                ):
                    callback(self, it, time.perf_counter() - self._t0)

                need_elbo = (
                    tol is not None
                    or (elbo_every and it % elbo_every == 0)
                    or it == max_iter
                )

                if need_elbo:
                    L, kp = self.compute_elbo(elbo_idx)
                    final_L = L

                    should_store = (
                        it == max_iter
                        or (elbo_every and it % elbo_every == 0)
                    )

                    if should_store:
                        self.elbo_history.append(
                            (it, L, time.perf_counter() - self._t0)
                        )

                    if verbose and should_store:
                        print(
                            f"iter {it}: ELBO {L:.6f}  "
                            f"E[K+] {kp:.2f}  "
                            f"MAP K {self.kappas[np.argmax(self.pi)]}  "
                            f"E[alpha] {self.alpha_bar:.4f}  "
                            f"active {self.active.sum()}"
                        )

                    if (
                        tol is not None
                        and prev_L is not None
                        and it >= min_iter
                        and not pruned_this_iter
                        and abs(L - prev_L) < tol * max(1.0, abs(prev_L))
                    ):
                        if verbose:
                            print(
                                f"restart {restart + 1}: converged at "
                                f"iter {it}; |ΔELBO|={abs(L - prev_L):.3e}"
                            )
                        break

                    prev_L = L

            if not np.isfinite(final_L):
                final_L, _ = self.compute_elbo(elbo_idx)
                self.elbo_history.append(
                    (it, final_L, time.perf_counter() - self._t0)
                )

            if final_L > best:
                best = final_L
                best_state = self._save_state()

        self._restore_state(best_state)
        self._finalize_responsibilities()

        return self
        
# ========================================================
# Composed models
# ========================================================
SVI_MFM_BetaBernoulli = type("SVI_MFM_BetaBernoulli", (BetaBernoulliAtoms, _MFMCore), {})
SVI_MFM_Wishart = type("SVI_MFM_Wishart", (ZeroMeanWishartAtoms, _MFMCore), {})
SVI_MFM_Gaussian = type("SVI_MFM_Gaussian", (NormalWishartAtoms, _MFMCore), {})
SVI_MFM_StudentT = type("SVI_MFM_StudentT", (StudentTAtoms, _MFMCore), {})

CAVI_MFM_BetaBernoulli = type(
    "CAVI_MFM_BetaBernoulli",
    (BetaBernoulliAtoms, _CAVIMFMCore),
    {},
)

CAVI_MFM_Wishart = type(
    "CAVI_MFM_Wishart",
    (ZeroMeanWishartAtoms, _CAVIMFMCore),
    {},
)

CAVI_MFM_Gaussian = type(
    "CAVI_MFM_Gaussian",
    (NormalWishartAtoms, _CAVIMFMCore),
    {},
)

CAVI_MFM_StudentT = type(
    "CAVI_MFM_StudentT",
    (StudentTAtoms, _CAVIMFMCore),
    {},
)

__all__ = [
    "_MFMCore",
    "_CAVIMFMCore",
    "SVI_MFM_BetaBernoulli",
    "SVI_MFM_Wishart",
    "SVI_MFM_Gaussian",
    "SVI_MFM_StudentT",
    "CAVI_MFM_BetaBernoulli",
    "CAVI_MFM_Wishart",
    "CAVI_MFM_Gaussian",
    "CAVI_MFM_StudentT",
]