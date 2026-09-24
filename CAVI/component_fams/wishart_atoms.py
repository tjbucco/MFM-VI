"""Zero-mean Wishart components for the MFM model.

    y_i | z_i = k  ~  N(0, Omega_k^{-1})
    Omega_k      ~  W(nu0, B0^{-1})

q(Omega_k) = W(nu_hat_k, B_hat_k^{-1})

This is the zero-mean Wishart family from the original CAVI_MFM code. The
components have no location parameter and can only distinguish clusters by
their scale about the origin. This is appropriate for data that is already
centered or when clustering should be based solely on covariance structure.

Parameterisation
----------------
Atoms are stored as blended expected sufficient statistics:

    Nk_hat (T,)      S_hat (T, D, D)
      = sum_i r_ik     = sum_i r_ik y_i y_i^T

and the natural parameters are derived from them:

    nu_k = nu0 + Nk
    B_k  = B0 + S_k

The SVI global step blends these sufficient statistics directly, which is
correct for the natural gradient update.
"""
import numpy as np
from scipy.special import psi, gammaln, multigammaln

__all__ = ["ZeroMeanWishartAtoms"]

class ZeroMeanWishartAtoms:
    """Drop-in replacement for other atom families in the MFM framework."""

    _atom_params = ("nu_hat", "B_hat")
    _family = "zeromean_wishart"

    # ---- setup ----
    def _setup_atoms(self, nu0=None, B0=None):
        """Initialize the Wishart prior.

        Parameters
        ----------
        nu0 : float, optional
            Degrees of freedom for the Wishart prior. Default is p + 4.
        B0 : array-like, shape (p, p), optional
            Scale matrix for the Wishart prior. If None, uses an empirical
            estimate based on the data covariance.
        """
        D = self.p
        self.nu0 = float(D + 4 if nu0 is None else nu0)
        if self.nu0 <= D - 1:
            raise ValueError(f"nu0 must exceed p - 1 = {D - 1}")

        if B0 is None:
            # Default: empirical covariance scaled by (nu0 - p - 1)
            emp_cov = np.atleast_2d(np.cov(self.Y, rowvar=False))
            self.B0 = emp_cov * (self.nu0 - D - 1)
        else:
            self.B0 = np.asarray(B0, float)

        self.logdet_B0 = np.linalg.slogdet(self.B0)[1]
        self._atoms_stale = True

    def _invalidate_atoms(self):
        """Mark cached expectations as stale."""
        self._atoms_stale = True
        self._B_hat_inv = None
        self._logdet_B_hat = None
        self._E_Omega_inv = None
        self._psi_p_nu_hat = None
        self._E_log_det_Omega = None

    # ---- derived quantities ----
    def _compute_atom_expectations(self):
        """Compute E_q[Omega_k] and E_q[log p(y_n | Omega_k)]."""
        if not self._atoms_stale:
            return

        D = self.p
        i = np.arange(1, D + 1)

        self._B_hat_inv = np.linalg.inv(self.B_hat)
        self._logdet_B_hat = np.linalg.slogdet(self.B_hat)[1]
        self._E_Omega_inv = self.nu_hat[:, None, None] * self._B_hat_inv
        self._psi_p_nu_hat = np.sum(
            psi(0.5 * (self.nu_hat[:, None] + 1 - i[None, :])), axis=1)
        self._E_log_det_Omega = (self._logdet_B_hat - D * np.log(2)
                                - self._psi_p_nu_hat)
        self._atoms_stale = False

    def _maha(self, Yb):
        """Compute Mahalanobis distance y^T Omega_k y for each atom.

        Parameters
        ----------
        Yb : array-like, shape (n, p)
            Data points to compute distances for.

        Returns
        -------
        maha : array, shape (n, T)
            Mahalanobis distances for each data point and atom.
        """
        self._compute_atom_expectations()
        return np.einsum('ni,kij,nj->nk', Yb, self._E_Omega_inv, Yb)

    # ---- likelihood ----
    def _expected_log_lik(self, Yb):
        """Compute E_q[log p(Y | Omega)] for each atom.

        Parameters
        ----------
        Yb : array-like, shape (n, p)
            Data points to compute likelihood for.

        Returns
        -------
        log_lik : array, shape (n, T)
            Expected log likelihood for each data point and atom.
        """
        self._compute_atom_expectations()
        D = self.p
        maha = self._maha(Yb)
        return -0.5 * (
            D * np.log(2 * np.pi) + self._E_log_det_Omega[None, :] + maha)

    # ---- global step ----
    def _atom_stats(self, Yb, w, scale):
        """Compute sufficient statistics from data and responsibilities.

        Parameters
        ----------
        Yb : array-like, shape (n, p)
            Data points.
        w : array-like, shape (n, T)
            Responsibilities for each data point and atom.
        scale : float
            Scaling factor for the sufficient statistics.

        Returns
        -------
        Nk : array, shape (T,)
            Sum of responsibilities for each atom.
        S : array, shape (T, p, p)
            Weighted scatter matrix for each atom.
        """
        Nk = scale * w.sum(axis=0)
        S = scale * np.einsum('nk,ni,nj->kij', w, Yb, Yb, optimize=True)
        return Nk, S

    def _blend_atoms(self, rho, Yb, w, scale):
        """Stochastic update of atom parameters.

        Parameters
        ----------
        rho : float
            Learning rate for the update.
        Yb : array-like, shape (n, p)
            Data points in the mini-batch.
        w : array-like, shape (n, T)
            Responsibilities for each data point and atom.
        scale : float
            Scaling factor for the sufficient statistics.
        """
        Nk_c, S_c = self._atom_stats(Yb, w, scale)
        self.nu_hat = (1 - rho) * self.nu_hat + rho * (self.nu0 + Nk_c)
        self.B_hat = (1 - rho) * self.B_hat + rho * (self.B0 + S_c)
        self._invalidate_atoms()

    # ---- KL divergence ----
    def _kl_atoms(self):
        """Compute KL divergence for each atom.

        Returns
        -------
        kl : array, shape (T,)
            KL divergence for each atom.
        """
        self._compute_atom_expectations()
        D = self.p
        nu_q, nu_p = self.nu_hat, self.nu0

        # KL(Wishart(nu_q, B_q^{-1}) || Wishart(nu_p, B0^{-1}))
        tr = np.einsum('ij,kji->k', self.B0, self._B_hat_inv)
        kl = (0.5 * nu_p * (self._logdet_B_hat - self.logdet_B0)
              + 0.5 * nu_q * (tr - D)
              + multigammaln(0.5 * nu_p, D)
              - multigammaln(0.5 * nu_q, D)
              + 0.5 * (nu_q - nu_p) * self._psi_p_nu_hat)
        return kl

    # ---- posterior predictive ----
    def _predict_log_lik(self, Y_new, weights, eps=1e-12):
        """Compute predictive log likelihood for new data.

        Parameters
        ----------
        Y_new : array-like, shape (n, p)
            New data points.
        weights : array-like, shape (T,)
            Mixture weights for each atom.
        eps : float, optional
            Small value to avoid log(0).

        Returns
        -------
        log_lik : float
            Average predictive log likelihood.
        """
        from scipy.special import logsumexp
        self._compute_atom_expectations()
        D = self.p
        w = np.maximum(np.asarray(weights, float), eps)
        w = w / w.sum()

        maha = self._maha(Y_new)
        log_lik = -0.5 * (
            D * np.log(2 * np.pi) + self._E_log_det_Omega[None, :] + maha)
        log_lik += np.log(w)[None, :]
        return float(np.mean(logsumexp(log_lik, axis=1)))

    # ---- summaries ----
    def component_precision_matrices(self):
        """Return E[Omega_k] for each component."""
        self._compute_atom_expectations()
        return self._E_Omega_inv.copy()

    def component_covariances(self):
        """Return E[Sigma_k] = E[Omega_k^{-1}] for each component."""
        self._compute_atom_expectations()
        denom = np.maximum(self.nu_hat - self.p - 1.0, 1e-6)
        return self._B_hat_inv / denom[:, None, None]

    # ---- initialization ----
    def _atom_init(self, Y0, scale, random_state, mode="kmeans"):
        """Initialize atoms from data.

        Parameters
        ----------
        Y0 : array-like, shape (n, p)
            Data to initialize from.
        scale : float
            Scaling factor for the sufficient statistics.
        random_state : int or RandomState
            Random seed for initialization.
        mode : str, optional
            Initialization method ("kmeans" or "random").
        """
        from sklearn.cluster import KMeans

        if mode == "kmeans":
            n_clust = min(self.T, Y0.shape[0])
            km = KMeans(n_clusters=n_clust, n_init=10, random_state=random_state)
            labels = km.fit_predict(Y0)

            # Sort by decreasing occupancy
            unique_labels, counts = np.unique(labels, return_counts=True)
            order = np.argsort(-counts)
            remap = np.empty(n_clust, dtype=int)
            for new_k, old_k in enumerate(order):
                remap[old_k] = new_k
            sorted_labels = remap[labels]

            for k in range(n_clust):
                Yk = Y0[sorted_labels == k]
                if Yk.shape[0] == 0:
                    continue
                Sk = (Yk.T @ Yk)  # Zero-mean sufficient statistic
                self._set_wishart(k, Yk.shape[0], Sk)
        else:
            # Random initialization
            rng = np.random.default_rng(random_state)
            idx = rng.choice(Y0.shape[0], size=self.T, replace=Y0.shape[0] < self.T)
            n_eff = max(2.0, 0.01 * scale * Y0.shape[0] / self.T)
            M = Y0[idx]
            cov = np.cov(Y0.T) + 1e-6 * np.eye(self.p)
            self.nu_hat = np.full(self.T, self.nu0 + n_eff)
            self.B_hat = np.tile(self.B0, (self.T, 1, 1)) + n_eff * (
                np.einsum('ki,kj->kij', M, M) + cov[None])

        self._invalidate_atoms()

    def _set_wishart(self, k, Nk, Sk):
        """Set Wishart parameters for atom k given sufficient statistics."""
        self.nu_hat[k] = self.nu0 + Nk
        self.B_hat[k] = self.B0 + Sk