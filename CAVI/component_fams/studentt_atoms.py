"""Multivariate Student-t components (Gaussian scale mixture).

    y_i | z_i = k, u_i ~ N(mu_k, (u_i Lambda_k)^{-1})
    u_i | z_i = k     ~ Gamma(df/2, df/2)
    (mu_k, Lambda_k)  ~ Normal-Wishart, as in gaussian_atoms.py

Marginalising u_i gives a multivariate t with `df` degrees of freedom.  This
is the standard fix for the over-splitting seen with Gaussian components on
cytometry: real cell populations are skewed and heavy-tailed, so a Gaussian
mixture spends extra components lining the tails of a single population
instead of modelling it with one.  flowClust / FLAME use t (and skew-t)
components for exactly this reason.

Collapsed local bound
---------------------
With q(u_i | z_i = k) = Gamma((df + D)/2, (df + Delta_ik)/2) at its optimum,
the u-terms of the ELBO collapse and the per-component contribution is

    L_ik = 0.5 E[log|Lambda_k|] - (D/2) log(pi df)
           + gammaln((df + D)/2) - gammaln(df/2)
           - ((df + D)/2) log(1 + Delta_ik / df)

    Delta_ik = E_q[(y_i - mu_k)^T Lambda_k (y_i - mu_k)]
             = D / beta_k + nu_k (y_i - m_k)^T C_k^{-1} (y_i - m_k)

i.e. a Student-t log density with E[log|Lambda|] in place of log|Lambda| and
Delta in place of the Mahalanobis distance.  As df -> infinity this reduces
to the Gaussian expected log-likelihood, which is the check in
`test_gaussian_limit` below.

Because L_ik already contains E[log p(u)] + H[q(u)], the ELBO is assembled
exactly as for the Gaussian family and the atom KL is the unchanged
Normal-Wishart KL.

Sufficient statistics
---------------------
Four accumulators, all linear in the data given responsibilities (so the SVI
natural-gradient blend remains a plain convex combination):

    Nk = sum_i r_ik                    (counts)
    Mk = sum_i r_ik E[u_ik]            (u-weighted counts)
    s  = sum_i r_ik E[u_ik] y_i
    S  = sum_i r_ik E[u_ik] y_i y_i^T

    beta_k = beta0 + Mk       nu_k = nu0 + Nk
    m_k    = (beta0 m0 + s) / beta_k
    C_k    = B0 + S + beta0 m0 m0^T - beta_k m_k m_k^T

Note the asymmetry: the Wishart degrees of freedom increment by the *count*
(each observation supplies one degree of freedom regardless of its scale),
while the location and scatter are down-weighted by E[u_ik].  Using Mk in
both places is a common and silent error -- it makes outlying points shrink
the Wishart df and inflates the covariances.

    E[u_ik] = (df + D) / (df + Delta_ik)

which is the robustness mechanism: a point far from component k in Mahalanobis
terms gets a small u and contributes little to that component's location and
scatter, without being excluded.
"""
import numpy as np
from scipy.special import gammaln, logsumexp

from .gaussian_atoms import NormalWishartAtoms, init_gaussian_atoms

__all__ = ["StudentTAtoms"]


class StudentTAtoms(NormalWishartAtoms):

    _atom_params = ("Nk_hat", "Mk_hat", "s_hat", "S_hat")
    _family = "studentt"

    # ---- setup ----
    def _setup_atoms(self, nu0=None, sF=1.0, beta0=0.01, m0=None, df=4.0):
        super()._setup_atoms(nu0=nu0, sF=sF, beta0=beta0, m0=m0)
        if df <= 2.0:
            raise ValueError("df must exceed 2 for a finite covariance")
        self.df = float(df)

    # ---- derived quantities ----
    def _compute_atom_expectations(self):
        if not self._atoms_stale:
            return
        D = self.p
        from scipy.special import psi
        self.beta_hat = self.beta0 + self.Mk_hat        # u-weighted
        self.nu_hat = self.nu0 + self.Nk_hat            # counts
        self.m_hat = ((self.beta0 * self.m0[None, :] + self.s_hat)
                      / self.beta_hat[:, None])

        C = (self.B0[None] + self.S_hat + self._m0outer[None]
             - self.beta_hat[:, None, None]
             * np.einsum('ki,kj->kij', self.m_hat, self.m_hat))
        C = 0.5 * (C + np.transpose(C, (0, 2, 1)))
        C += 1e-8 * np.eye(D)[None]
        self.C_hat = C
        self.C_inv = np.linalg.inv(C)
        self.logdet_C = np.linalg.slogdet(C)[1]

        i_vec = np.arange(1, D + 1)
        self.psi_p_nu = np.sum(
            psi(0.5 * (self.nu_hat[:, None] + 1 - i_vec[None, :])), axis=1)
        self.E_log_det_Lambda = (self.psi_p_nu + D * np.log(2.0)
                                 - self.logdet_C)
        self._atoms_stale = False

    def _delta(self, Yb):
        """Delta_ik = D / beta_k + nu_k * maha_ik, shape (n, T)."""
        self._compute_atom_expectations()
        return (self.p / self.beta_hat[None, :]
                + self.nu_hat[None, :] * self._maha(Yb))

    def _E_u(self, Delta):
        return (self.df + self.p) / (self.df + Delta)

    # ---- likelihood ----
    def _expected_log_lik(self, Yb):
        D, df = self.p, self.df
        Delta = self._delta(Yb)
        return (0.5 * self.E_log_det_Lambda[None, :]
                - 0.5 * D * np.log(np.pi * df)
                + gammaln(0.5 * (df + D)) - gammaln(0.5 * df)
                - 0.5 * (df + D) * np.log1p(Delta / df))

    # ---- global step ----
    def _atom_stats(self, Yb, w, scale):
        Eu = self._E_u(self._delta(Yb))                  # (n, T)
        wu = w * Eu
        Nk = scale * w.sum(axis=0)
        Mk = scale * wu.sum(axis=0)
        s = scale * (Yb.T @ wu).T
        S = scale * np.einsum('nk,ni,nj->kij', wu, Yb, Yb, optimize=True)
        return Nk, Mk, s, S

    def _blend_atoms(self, rho, Yb, w, scale):
        Nk_c, Mk_c, s_c, S_c = self._atom_stats(Yb, w, scale)
        self.Nk_hat = (1 - rho) * self.Nk_hat + rho * Nk_c
        self.Mk_hat = (1 - rho) * self.Mk_hat + rho * Mk_c
        self.s_hat = (1 - rho) * self.s_hat + rho * s_c
        self.S_hat = (1 - rho) * self.S_hat + rho * S_c
        self._invalidate_atoms()

    # ---- predictive ----
    def _predict_log_lik(self, Y_new, weights, eps=1e-12):
        """Plug-in multivariate t, NOT the exact posterior predictive.

        Unlike the Gaussian family (where marginalising the Normal-Wishart
        gives a t in closed form), there is no closed form here once the
        scale mixture is added, so this plugs in E[Sigma_k] and the fixed df.
        Comparable across methods sharing this family; do not compare it
        against the Gaussian family's exact predictive."""
        self._compute_atom_expectations()
        D, df = self.p, self.df
        Sig = self.component_covariances()
        P = np.linalg.inv(Sig)
        logdet_P = -np.linalg.slogdet(Sig)[1]
        w = np.maximum(np.asarray(weights, float), 0.0)
        w = w / w.sum()

        maha = np.empty((len(Y_new), self.T))
        for k in range(self.T):
            d = Y_new - self.m_hat[k]
            maha[:, k] = np.einsum('ni,ni->n', d @ P[k], d)
        lp = (gammaln(0.5 * (df + D)) - gammaln(0.5 * df)
              - 0.5 * D * np.log(df * np.pi)
              + 0.5 * logdet_P[None, :]
              - 0.5 * (df + D) * np.log1p(maha / df)
              + np.log(np.maximum(w, eps))[None, :])
        return float(np.mean(logsumexp(lp, axis=1)))

    def component_covariances(self):
        """E[Sigma_k] of the Gaussian kernel (not of the t marginal)."""
        self._compute_atom_expectations()
        denom = np.maximum(self.nu_hat - self.p - 1.0, 1e-6)
        return self.C_hat / denom[:, None, None]

    # ---- init ----
    def _atom_init(self, Y0, scale, random_state, mode="kmeans"):
        if mode == "random":
            rng = np.random.default_rng(random_state)
            idx = rng.choice(Y0.shape[0], size=self.T,
                             replace=Y0.shape[0] < self.T)
            n_eff = max(2.0, 0.01 * scale * Y0.shape[0] / self.T)
            M = Y0[idx]
            cov = np.cov(Y0.T) + 1e-6 * np.eye(self.p)
            self.Nk_hat = np.full(self.T, n_eff)
            self.Mk_hat = np.full(self.T, n_eff)
            self.s_hat = n_eff * M
            self.S_hat = n_eff * (np.einsum('ki,kj->kij', M, M) + cov[None])
        else:
            Nk, s, S = init_gaussian_atoms(Y0, self.T, scale, random_state)
            self.Nk_hat, self.Mk_hat, self.s_hat, self.S_hat = Nk, Nk, s, S
        self._invalidate_atoms()
