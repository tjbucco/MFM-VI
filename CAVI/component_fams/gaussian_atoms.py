"""Gaussian components with unknown mean and precision (Normal-Wishart).

    y_i | z_i = k  ~  N(mu_k, Lambda_k^{-1})
    Lambda_k       ~  W(nu0, B0^{-1})
    mu_k | Lambda_k~  N(m0, (beta0 Lambda_k)^{-1})

q(mu_k, Lambda_k) = N(mu_k | m_k, (beta_k Lambda_k)^{-1}) W(Lambda_k | nu_k, C_k^{-1})

This is the zero-mean Wishart family of the original code with the mean freed.
Zero-mean components can only distinguish clusters by their scale about the
origin, which is wrong for anything whose groups differ in location -- digits,
flow cytometry, essentially any real clustering problem.

Parameterisation
----------------
Atoms are stored as *blended expected sufficient statistics*

    Nk_hat (T,)      s_hat (T, D)      S_hat (T, D, D)
      = sum_i r_ik     = sum_i r_ik y_i  = sum_i r_ik y_i y_i^T

and the natural parameters are derived from them:

    beta_k = beta0 + Nk          nu_k = nu0 + Nk
    m_k    = (beta0 m0 + s_k) / beta_k
    C_k    = B0 + S_k + beta0 m0 m0^T - beta_k m_k m_k^T

The last identity is the usual
    C_k = B0 + N_k S_k^{emp} + (beta0 N_k / (beta0 + N_k))(xbar - m0)(xbar - m0)^T
rewritten so that it is affine in (Nk, s, S).  That matters: the SVI global
step is a natural-gradient step, which is a convex combination in expected
sufficient statistic space, so blending (Nk, s, S) is exactly correct and
needs no special handling.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

import numpy as np
from scipy.special import psi, gammaln, multigammaln
from sklearn.cluster import KMeans

__all__ = ["NormalWishartAtoms", "init_gaussian_atoms"]


class NormalWishartAtoms:
    """Drop-in replacement for BetaBernoulliAtoms."""

    _atom_params = ("Nk_hat", "s_hat", "S_hat")
    _family = "gaussian"

    # ---- setup ----
    def _setup_atoms(self, nu0=None, sF=1.0, beta0=0.01, m0=None,
                     B0_mode="identity"):
        """B0_mode:
            "identity" -- B0 = sF * I   (scale depends on how you scaled X)
            "cov"      -- B0 = sF * Cov(X), m0 = mean(X) unless given.
                          Makes the prior affine-equivariant, so results no
                          longer depend on per-marker standardisation."""
        D = self.p
        self.nu0 = float(D + 2 if nu0 is None else nu0)
        if self.nu0 <= D - 1:
            raise ValueError(f"nu0 must exceed p - 1 = {D - 1}")
        self.sF = float(sF)
        self.B0_mode = B0_mode
        if B0_mode == "cavi_mfm":
            self.nu0 = float(D + 4 if nu0 is None else nu0)
            emp_cov = np.atleast_2d(np.cov(self.Y, rowvar=False))
            self.B0 = emp_cov * (self.nu0 - D - 1.0)
        elif B0_mode == "cov":
            C = np.cov(self.Y.T) + 1e-8 * np.eye(D)
            self.B0 = self.sF * C
            if m0 is None:
                m0 = self.Y.mean(axis=0)
        else:
            self.B0 = self.sF * np.eye(D)
        self.beta0 = float(beta0)
        self.m0 = np.zeros(D) if m0 is None else np.asarray(m0, float)
        self.m0 = self.Y.mean(axis=0) if m0 is None else np.asarray(m0, float)
        self.logdet_B0 = np.linalg.slogdet(self.B0)[1]
        self._m0outer = self.beta0 * np.outer(self.m0, self.m0)
        self._atoms_stale = True

    def _invalidate_atoms(self):
        self._atoms_stale = True

    # ---- derived quantities ----
    def _compute_atom_expectations(self):
        if not self._atoms_stale:
            return
        D = self.p
        Nk, s, S = self.Nk_hat, self.s_hat, self.S_hat

        self.beta_hat = self.beta0 + Nk
        self.nu_hat = self.nu0 + Nk
        self.m_hat = (self.beta0 * self.m0[None, :] + s) / self.beta_hat[:, None]

        C = (self.B0[None] + S + self._m0outer[None]
             - self.beta_hat[:, None, None]
             * np.einsum('ki,kj->kij', self.m_hat, self.m_hat))
        C = 0.5 * (C + np.transpose(C, (0, 2, 1)))          # symmetrise
        C += 1e-8 * np.eye(D)[None]                          # ridge
        self.C_hat = C
        self.C_inv = np.linalg.inv(C)
        self.logdet_C = np.linalg.slogdet(C)[1]

        i_vec = np.arange(1, D + 1)
        self.psi_p_nu = np.sum(
            psi(0.5 * (self.nu_hat[:, None] + 1 - i_vec[None, :])), axis=1)
        self.E_log_det_Lambda = (self.psi_p_nu + D * np.log(2.0)
                                 - self.logdet_C)
        self._atoms_stale = False

    def _maha(self, Yb, scaled=True):
        """(y - m_k)^T C_k^{-1} (y - m_k), shape (n, T).

        Vectorised when the (n, T, D) buffer is small enough (< ~200 MB);
        otherwise looped over components to keep memory at (n, D)."""
        n = Yb.shape[0]
        if n * self.T * self.p * 8 < 2e8:
            d = Yb[:, None, :] - self.m_hat[None, :, :]          # (n, T, D)
            return np.einsum('nki,kij,nkj->nk', d, self.C_inv, d,
                             optimize=True)
        out = np.empty((n, self.T))
        for k in range(self.T):
            d = Yb - self.m_hat[k]
            out[:, k] = np.einsum('ni,ni->n', d @ self.C_inv[k], d)
        return out

    # ---- likelihood ----
    def _expected_log_lik(self, Yb):
        self._compute_atom_expectations()
        D = self.p
        maha = self._maha(Yb)
        return 0.5 * (self.E_log_det_Lambda[None, :]
                      - D * np.log(2 * np.pi)
                      - D / self.beta_hat[None, :]
                      - self.nu_hat[None, :] * maha)

    # ---- global step ----
    def _atom_stats(self, Yb, w, scale):
        Nk = scale * w.sum(axis=0)
        s = scale * (Yb.T @ w).T
        S = scale * np.einsum('nk,ni,nj->kij', w, Yb, Yb, optimize=True)
        return Nk, s, S

    def _blend_atoms(self, rho, Yb, w, scale):
        Nk_c, s_c, S_c = self._atom_stats(Yb, w, scale)
        self.Nk_hat = (1 - rho) * self.Nk_hat + rho * Nk_c
        self.s_hat = (1 - rho) * self.s_hat + rho * s_c
        self.S_hat = (1 - rho) * self.S_hat + rho * S_c
        self._invalidate_atoms()

    # ---- KL(q || prior), one value per atom ----
    def _kl_atoms(self):
        self._compute_atom_expectations()
        D = self.p
        nu_q, nu_p = self.nu_hat, self.nu0
        beta_q, beta_p = self.beta_hat, self.beta0
        E_logdet = self.E_log_det_Lambda

        # --- Gaussian part: E_q[ log q(mu|Lambda) - log p(mu|Lambda) ] ---
        dm = self.m_hat - self.m0[None, :]
        maha_m = np.einsum('ki,kij,kj->k', dm, self.C_inv, dm)
        kl_mu = 0.5 * (D * (np.log(beta_q) - np.log(beta_p))
                       - D * (1.0 - beta_p / beta_q)
                       + beta_p * nu_q * maha_m)

        # --- Wishart part: KL( W(nu_q, C_q^-1) || W(nu_p, B0^-1) ) ---
        tr = np.einsum('ij,kji->k', self.B0, self.C_inv)
        logB_q = (0.5 * nu_q * self.logdet_C
                  - 0.5 * nu_q * D * np.log(2.0)
                  - multigammaln(0.5 * nu_q, D))
        logB_p = (-0.5 * nu_p * self.logdet_B0
                  - 0.5 * nu_p * D * np.log(2.0)
                  - multigammaln(0.5 * nu_p, D))
        kl_lam = (logB_q - logB_p
                  + 0.5 * (nu_q - nu_p) * E_logdet
                  + 0.5 * nu_q * (tr - D))
        return kl_mu + kl_lam

    # ---- posterior predictive: multivariate t ----
    def _predict_log_lik(self, Y_new, weights, eps=1e-12):
        """log sum_k w_k St(y | m_k, L_k, df_k), averaged over rows.

        Exact under q: marginalising the Normal-Wishart gives a Student-t,
        so this is the true variational posterior predictive, not a plug-in.
        """
        from scipy.special import logsumexp
        self._compute_atom_expectations()
        D = self.p
        Y_new = np.asarray(Y_new, float)
        df = self.nu_hat - D + 1.0
        fac = (self.beta_hat * df) / (self.beta_hat + 1.0)   # L_k = fac * C^-1
        w = np.maximum(np.asarray(weights, float), 0.0)
        w = w / w.sum()

        maha = self._maha(Y_new) * fac[None, :]
        logdet_L = D * np.log(fac) - self.logdet_C
        lp = (gammaln(0.5 * (df + D))[None, :] - gammaln(0.5 * df)[None, :]
              - 0.5 * D * np.log(df * np.pi)[None, :]
              + 0.5 * logdet_L[None, :]
              - 0.5 * (df + D)[None, :] * np.log1p(maha / df[None, :])
              + np.log(np.maximum(w, eps))[None, :])
        return float(np.mean(logsumexp(lp, axis=1)))

    # ---- summaries ----
    def component_means(self):
        self._compute_atom_expectations()
        return self.m_hat.copy()

    def component_covariances(self):
        """E[Sigma_k] = C_k / (nu_k - D - 1)."""
        self._compute_atom_expectations()
        denom = np.maximum(self.nu_hat - self.p - 1.0, 1e-6)
        return self.C_hat / denom[:, None, None]

    # ---- initialisation ----
    def _atom_init(self, Y0, scale, random_state, mode="kmeans"):
        if mode == "random":
            rng = np.random.default_rng(random_state)
            idx = rng.choice(Y0.shape[0], size=self.T,
                             replace=Y0.shape[0] < self.T)
            n_eff = max(2.0, 0.01 * scale * Y0.shape[0] / self.T)
            M = Y0[idx]
            self.Nk_hat = np.full(self.T, n_eff)
            self.s_hat = n_eff * M
            cov = np.cov(Y0.T) + 1e-6 * np.eye(self.p)
            self.S_hat = n_eff * (np.einsum('ki,kj->kij', M, M) + cov[None])
        elif mode == "kmeans_cavi":
            N, D = Y0.shape
            n_init = min(self.T, N)
            
            # 1. Run K-Means
            kmeans = KMeans(n_clusters=n_init, n_init=10, random_state=random_state)
            labels = kmeans.fit_predict(Y0)
            
            # 2. Allocate sufficient statistics buffers
            Nk = np.zeros(self.T)
            s = np.zeros((self.T, D))
            S = np.zeros((self.T, D, D))  # Uncentered sum of squares: \sum y_i y_i^T
            
            for k in range(n_init):
                Yk = Y0[labels == k]
                nk = Yk.shape[0]
                if nk == 0:
                    continue  # Keep as 0 so parameters revert to prior in _compute_atom_expectations
                    
                Nk[k] = nk
                s[k] = Yk.sum(axis=0)
                S[k] = Yk.T @ Yk  # MUST BE UNCENTERED for _compute_atom_expectations
                
            self.Nk_hat = Nk
            self.s_hat = s
            self.S_hat = S
    
        else:
            self.Nk_hat, self.s_hat, self.S_hat = init_gaussian_atoms(
                Y0, self.T, scale, random_state)
        self._invalidate_atoms()


# =====================================================================
def _kmeanspp(Y, T, rng):
    n = Y.shape[0]
    idx = [int(rng.integers(n))]
    d2 = np.sum((Y - Y[idx[0]]) ** 2, axis=1)
    for _ in range(1, T):
        tot = d2.sum()
        j = int(rng.integers(n)) if tot <= 0 else int(rng.choice(n, p=d2 / tot))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Y - Y[j]) ** 2, axis=1))
    return Y[idx].copy()


def init_gaussian_atoms(Y0, T, scale, random_state, n_seeds=5, n_refine=25):
    """k-means initialisation returning blended sufficient statistics."""
    Y0 = np.asarray(Y0, float)
    n, D = Y0.shape
    rng = np.random.default_rng(random_state)
    best, best_obj = None, np.inf
    sq = np.sum(Y0 ** 2, axis=1)

    for _ in range(n_seeds):
        M = _kmeanspp(Y0, T, rng)
        labels = None
        for _ in range(n_refine):
            d2 = sq[:, None] - 2.0 * (Y0 @ M.T) + np.sum(M ** 2, axis=1)[None]
            new = np.argmin(d2, axis=1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for k in range(T):
                m = labels == k
                if m.sum() >= 1:
                    M[k] = Y0[m].mean(axis=0)
                else:                       # reseed the emptiest cluster
                    M[k] = Y0[int(rng.integers(n))]
        obj = float(np.min(d2, axis=1).sum())
        if obj < best_obj:
            best_obj, best = obj, (labels.copy(), M.copy())

    labels, M = best
    Z = np.zeros((n, T))
    Z[np.arange(n), labels] = 1.0
    Nk = scale * Z.sum(axis=0)
    s = scale * (Y0.T @ Z).T
    S = scale * np.einsum('nk,ni,nj->kij', Z, Y0, Y0, optimize=True)

    # starved atoms: place at their centre with a weak pseudo-count so they
    # stay competitive instead of collapsing to the prior
    dead = Nk < 2.0
    if dead.any():
        w = max(2.0, 0.01 * scale * n / T)
        cov = np.cov(Y0.T) + 1e-6 * np.eye(D)
        Nk[dead] = w
        s[dead] = w * M[dead]
        S[dead] = w * (np.einsum('ki,kj->kij', M[dead], M[dead]) + cov[None])
    return Nk, s, S
