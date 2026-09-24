"""Shared Beta-Bernoulli component family.

This is the only model-specific part that differs from the Gaussian /
Wishart code: it replaces `_wishart_from_stats`, `_compute_atom_expectations`,
`_expected_log_lik`, the atom blend and the Wishart KL.  Every algorithm in
this package (dynamic MFM, DP, fixed-T) inherits it unchanged, so the three
methods share an identical likelihood and identical atom updates and differ
only in how they model the mixture weights.

Model
-----
    y_i | z_i = k ~ prod_d Bernoulli(beta_kd),   beta_kd ~ Beta(a0, b0)
    q(beta_kd)    = Beta(a_hat_kd, b_hat_kd)

E[log beta_kd]     = psi(a_kd) - psi(a_kd + b_kd)
E[log(1-beta_kd)]  = psi(b_kd) - psi(a_kd + b_kd)
"""
import numpy as np
from scipy.special import psi, gammaln

__all__ = ["BetaBernoulliAtoms", "sample_kplus"]


class BetaBernoulliAtoms:
    """Mixin.  Expects `self.a_hat`, `self.b_hat` of shape (T, D)."""

    _atom_params = ("a_hat", "b_hat")
    _family = "bernoulli"

    # ---- setup ----
    def _setup_atoms(self, a_beta0=1.0, b_beta0=1.0):
        self.a_beta0 = float(a_beta0)
        self.b_beta0 = float(b_beta0)
        self._atoms_stale = True

    def _invalidate_atoms(self):
        self._atoms_stale = True

    def _compute_atom_expectations(self):
        """Lazy: only recompute when (a_hat, b_hat) changed."""
        if not self._atoms_stale:
            return
        a, b = self.a_hat, self.b_hat
        psi_ab = psi(a + b)
        self.E_log_beta = psi(a) - psi_ab          # (T, D)
        self.E_log_1mbeta = psi(b) - psi_ab        # (T, D)
        # log-lik in one matmul:  Y @ slope + offset
        self._llik_slope = (self.E_log_beta - self.E_log_1mbeta).T   # (D, T)
        self._llik_offset = self.E_log_1mbeta.sum(axis=1)            # (T,)
        self._atoms_stale = False

    # ---- likelihood ----
    def _expected_log_lik(self, Yb):
        """E_q[log p(y_i | z_i = k)], shape (n, T)."""
        self._compute_atom_expectations()
        return Yb @ self._llik_slope + self._llik_offset[None, :]

    # ---- global step ----
    def _atom_stats(self, Yb, w, scale):
        """Intermediate natural parameters from weights w (n, T)."""
        S = scale * (Yb.T @ w).T                   # (T, D) weighted successes
        Nk = scale * w.sum(axis=0)                 # (T,)
        return self.a_beta0 + S, self.b_beta0 + Nk[:, None] - S

    def _blend_atoms(self, rho, Yb, w, scale):
        a_c, b_c = self._atom_stats(Yb, w, scale)
        self.a_hat = (1.0 - rho) * self.a_hat + rho * a_c
        self.b_hat = (1.0 - rho) * self.b_hat + rho * b_c
        self._invalidate_atoms()

    # ---- KL ----
    def _kl_atoms(self):
        """KL( Beta(a_hat, b_hat) || Beta(a0, b0) ) summed over pixels.

        Returns one value per atom, shape (T,)."""
        self._compute_atom_expectations()
        a, b = self.a_hat, self.b_hat
        a0, b0 = self.a_beta0, self.b_beta0
        logB_hat = gammaln(a) + gammaln(b) - gammaln(a + b)
        logB_0 = gammaln(a0) + gammaln(b0) - gammaln(a0 + b0)
        kl = (logB_0 - logB_hat
              + (a - a0) * self.E_log_beta
              + (b - b0) * self.E_log_1mbeta)
        return kl.sum(axis=1)

    # ---- posterior summaries of the atoms ----
    def beta_mean(self):
        """E_q[beta_k], shape (T, D).  Pixel means of each component."""
        return self.a_hat / (self.a_hat + self.b_hat)

    # ---- generic hooks (shared interface with NormalWishartAtoms) ----
    def _predict_log_lik(self, Y_new, weights, eps=1e-12):
        """Exact under mean field: q(pi) and q(beta) are independent and the
        pixels are independent given the component."""
        from scipy.special import logsumexp
        Y_new = np.asarray(Y_new, dtype=float)
        B = np.clip(self.beta_mean(), eps, 1.0 - eps)
        w = np.maximum(np.asarray(weights, dtype=float), 0.0)
        w = w / w.sum()
        lp = (Y_new @ np.log(B / (1.0 - B)).T
              + np.log1p(-B).sum(axis=1)[None, :]
              + np.log(np.maximum(w, eps))[None, :])
        return float(np.mean(logsumexp(lp, axis=1)))

    def _atom_init(self, Y0, scale, random_state, mode="kmeans"):
        from .init_utils import init_beta_atoms, init_beta_atoms_random
        if mode == "random":
            self.a_hat, self.b_hat, _ = init_beta_atoms_random(
                self.T, self.p, random_state)
        else:
            self.a_hat, self.b_hat, _ = init_beta_atoms(
                Y0, self.T, self.a_beta0, self.b_beta0, scale, random_state,
                n_seeds=5, n_refine=20)
        self._invalidate_atoms()


def sample_kplus(r, n_samples, rng, chunk=64):
    """Monte-Carlo draws of K_+ = #occupied components given q(z).

    r : (N, K) responsibilities.  Chunked over samples so the cost is one
    (chunk, N, K) broadcast rather than a Python loop over draws.
    """
    N, K = r.shape
    if n_samples == 0:
        return np.empty(0, dtype=int)
    cum = np.cumsum(r, axis=1)
    out = np.empty(n_samples, dtype=int)
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        u = rng.random((m, N, 1))
        Z = np.minimum((u > cum[None, :, :]).sum(axis=2), K - 1)   # (m, N)
        occ = np.zeros((m, K), dtype=bool)
        occ[np.repeat(np.arange(m), N), Z.ravel()] = True
        out[done:done + m] = occ.sum(axis=1)
        done += m
    return out
