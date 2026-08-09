import numpy as np
from scipy.special import psi, logsumexp

from .utils import niw_update, niw_expectations, mahalanobis_matrix

# ========================================================
# Baseline 1: Truncated stick-breaking CAVI for a DP mixture
#             (Blei & Jordan, 2006)
# ========================================================
class DPMixtureCAVI:
    """
    Standard mean-field CAVI for a Dirichlet Process Gaussian mixture,
    truncated at T components via the stick-breaking representation.

    q(V_k) = Beta(gamma_k1, gamma_k2),  k = 1,...,T-1   (V_T := 1)
    q(mu_k, Omega_k) = NIW(hat_b_k, hat_lambda_k, hat_B_k, hat_nu_k)
    q(Z_n) = Categorical(phi_n,:)

    alpha (DP concentration) itself has a Gamma(s1, s2) prior and is
    updated variationally (Blei & Jordan Section 4).
    """
    def __init__(self, Y, T, s1=1.0, s2=1.0, lambda0=1e-6, nu0=None, b0_mu=None, B0=None):
        self.Y = Y
        self.N, self.p = Y.shape
        self.T = T
        self.s1, self.s2 = s1, s2

        self.lambda0 = lambda0
        self.b0_mu = np.mean(Y, axis=0) if b0_mu is None else b0_mu
        self.nu0 = self.p + 2 if nu0 is None else nu0
        if B0 is None:
            emp_cov = np.cov(Y, rowvar=False)
            if self.p == 1:
                emp_cov = np.array([[emp_cov]])
            self.B0 = emp_cov * (self.nu0 - self.p - 1)
        else:
            self.B0 = B0

        self.gamma1 = np.ones(self.T - 1)
        self.gamma2 = np.ones(self.T - 1)

        self.s1_hat = s1
        self.s2_hat = s2

        self.phi = np.full((self.N, self.T), 1.0 / self.T)

        rng_local = np.random.default_rng(0)
        init_labels = rng_local.integers(0, self.T, size=self.N)
        phi_init = np.eye(self.T)[init_labels]
        self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k = niw_update(
            self.Y, phi_init, self.lambda0, self.nu0, self.b0_mu, self.B0
        )
        self._compute_expectations()

    def _compute_expectations(self):
        self.E_mu_k, self.E_Omega_k_inv, self.E_log_det_Omega_k, self.trace_term = niw_expectations(
            self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k, self.p
        )

    def _E_log_stick_weights(self):
        E_log_V = psi(self.gamma1) - psi(self.gamma1 + self.gamma2)
        E_log_1mV = psi(self.gamma2) - psi(self.gamma1 + self.gamma2)

        E_log_pi = np.zeros(self.T)
        cum_log_1mV = 0.0
        for k in range(self.T - 1):
            E_log_pi[k] = E_log_V[k] + cum_log_1mV
            cum_log_1mV += E_log_1mV[k]
        E_log_pi[self.T - 1] = cum_log_1mV
        return E_log_pi, E_log_1mV

    def _update_phi(self):
        E_log_pi, _ = self._E_log_stick_weights()
        E_maha = mahalanobis_matrix(self.Y, self.E_mu_k, self.E_Omega_k_inv, self.trace_term)

        log_rho = (E_log_pi[np.newaxis, :] - 0.5 * self.E_log_det_Omega_k[np.newaxis, :]
                    - 0.5 * E_maha)
        log_phi = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        self.phi = np.exp(log_phi)

    def _update_sticks(self):
        alpha_hat = self.s1_hat / self.s2_hat
        Nk = self.phi.sum(axis=0)

        for k in range(self.T - 1):
            self.gamma1[k] = 1.0 + Nk[k]
            self.gamma2[k] = alpha_hat + np.sum(Nk[k + 1:])

    def _update_alpha(self):
        _, E_log_1mV = self._E_log_stick_weights()
        self.s1_hat = self.s1 + (self.T - 1)
        self.s2_hat = self.s2 - np.sum(E_log_1mV)
        if self.s2_hat <= 0:
            self.s2_hat = 1e-6

    def _update_niw(self):
        self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k = niw_update(
            self.Y, self.phi, self.lambda0, self.nu0, self.b0_mu, self.B0
        )

    def fit(self, max_iter=100, tol=1e-5, verbose=False):
        for i in range(max_iter):
            phi_old = self.phi.copy()

            self._update_phi()
            self._update_sticks()
            self._update_alpha()
            self._update_niw()
            self._compute_expectations()

            change = np.max(np.abs(self.phi - phi_old))
            if verbose:
                print(f"[DP-CAVI] iter {i+1}: max resp change = {change:.6f}")
            if i > 0 and change < tol:
                break
        return self.phi

    def infer_K(self, threshold_frac=0.02, min_count=2):
        Nk = self.phi.sum(axis=0)
        thresh = max(min_count, threshold_frac * self.N)
        return int(np.sum(Nk >= thresh)), Nk


# ========================================================
# Baseline 2: CAVI for a fixed-T finite Bayesian Gaussian mixture
#             (symmetric Dirichlet(alpha0/T) prior, no model selection)
# ========================================================
class FiniteMixtureCAVI:
    """
    Standard mean-field CAVI for a finite Gaussian mixture with a FIXED
    number of components T. Prior on weights: eta ~ Dirichlet(alpha0/T,...).
    No posterior over K is computed -- K must be read off by inspecting
    which of the T components retain non-trivial weight/occupancy.
    """
    def __init__(self, Y, T, alpha0=1.0, lambda0=1e-6, nu0=None, b0_mu=None, B0=None):
        self.Y = Y
        self.N, self.p = Y.shape
        self.T = T
        self.alpha0 = alpha0

        self.lambda0 = lambda0
        self.b0_mu = np.mean(Y, axis=0) if b0_mu is None else b0_mu
        self.nu0 = self.p + 2 if nu0 is None else nu0
        if B0 is None:
            emp_cov = np.cov(Y, rowvar=False)
            if self.p == 1:
                emp_cov = np.array([[emp_cov]])
            self.B0 = emp_cov * (self.nu0 - self.p - 1)
        else:
            self.B0 = B0

        self.alpha_hat = np.full(self.T, alpha0 / self.T)
        self.phi = np.full((self.N, self.T), 1.0 / self.T)

        rng_local = np.random.default_rng(1)
        init_labels = rng_local.integers(0, self.T, size=self.N)
        phi_init = np.eye(self.T)[init_labels]
        self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k = niw_update(
            self.Y, phi_init, self.lambda0, self.nu0, self.b0_mu, self.B0
        )
        self._compute_expectations()

    def _compute_expectations(self):
        self.E_mu_k, self.E_Omega_k_inv, self.E_log_det_Omega_k, self.trace_term = niw_expectations(
            self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k, self.p
        )

    def _update_phi(self):
        E_log_eta = psi(self.alpha_hat) - psi(np.sum(self.alpha_hat))
        E_maha = mahalanobis_matrix(self.Y, self.E_mu_k, self.E_Omega_k_inv, self.trace_term)

        log_rho = (E_log_eta[np.newaxis, :] - 0.5 * self.E_log_det_Omega_k[np.newaxis, :]
                    - 0.5 * E_maha)
        log_phi = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        self.phi = np.exp(log_phi)

    def _update_eta(self):
        Nk = self.phi.sum(axis=0)
        self.alpha_hat = (self.alpha0 / self.T) + Nk

    def _update_niw(self):
        self.hat_lambda_k, self.hat_nu_k, self.hat_b_k, self.hat_B_k = niw_update(
            self.Y, self.phi, self.lambda0, self.nu0, self.b0_mu, self.B0
        )

    def fit(self, max_iter=100, tol=1e-5, verbose=False):
        for i in range(max_iter):
            phi_old = self.phi.copy()

            self._update_phi()
            self._update_eta()
            self._update_niw()
            self._compute_expectations()

            change = np.max(np.abs(self.phi - phi_old))
            if verbose:
                print(f"[Finite-T CAVI] iter {i+1}: max resp change = {change:.6f}")
            if i > 0 and change < tol:
                break
        return self.phi

    def infer_K(self, threshold_frac=0.02, min_count=2):
        Nk = self.phi.sum(axis=0)
        thresh = max(min_count, threshold_frac * self.N)
        return int(np.sum(Nk >= thresh)), Nk