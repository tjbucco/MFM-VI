import numpy as np
from scipy.special import psi

# ========================================================
# Shared NIW conjugate-update helpers
# ========================================================
def niw_update(Y, phi, lambda0, nu0, b0_mu, B0):
    """
    Given data Y (N x p) and responsibilities phi (N x T), compute updated
    NIW variational parameters for T components. Mirrors CAVI_MFM's
    _update_q_mu_Omega, but for a single fixed model (no marginalization
    over K).
    """
    N, p = Y.shape
    T = phi.shape[1]

    Nk = phi.sum(axis=0)
    safe_Nk = np.where(Nk > 1e-9, Nk, 1e-9)
    y_bar_k = (phi.T @ Y) / safe_Nk[:, np.newaxis]

    hat_lambda_k = np.zeros(T)
    hat_nu_k = np.zeros(T)
    hat_b_k = np.zeros((T, p))
    hat_B_k = np.zeros((T, p, p))

    for k in range(T):
        if Nk[k] < 1e-9:
            hat_lambda_k[k] = lambda0
            hat_nu_k[k] = nu0
            hat_b_k[k] = b0_mu
            hat_B_k[k] = B0
            continue

        y_centered = Y - y_bar_k[k]
        S_k = (y_centered.T * phi[:, k]) @ y_centered

        hat_lambda_k[k] = lambda0 + Nk[k]
        hat_nu_k[k] = nu0 + Nk[k]
        hat_b_k[k] = (lambda0 * b0_mu + Nk[k] * y_bar_k[k]) / hat_lambda_k[k]

        y_diff = y_bar_k[k] - b0_mu
        hat_B_k[k] = B0 + S_k + (lambda0 * Nk[k] / hat_lambda_k[k]) * np.outer(y_diff, y_diff)

    return hat_lambda_k, hat_nu_k, hat_b_k, hat_B_k

def niw_expectations(hat_lambda_k, hat_nu_k, hat_b_k, hat_B_k, p):
    """Compute E[mu_k], E[Omega_k^-1], E[log|Omega_k|], trace term, per component."""
    T = hat_lambda_k.shape[0]
    E_mu_k = hat_b_k.copy()
    E_Omega_k_inv = np.zeros((T, p, p))
    E_log_det_Omega_k = np.zeros(T)
    trace_term = np.zeros(T)

    for k in range(T):
        E_Omega_k_inv[k] = hat_nu_k[k] * np.linalg.inv(hat_B_k[k])
        log_det_B_k = np.linalg.slogdet(hat_B_k[k])[1]
        digamma_term = np.sum(psi(0.5 * (hat_nu_k[k] + 1 - np.arange(1, p + 1))))
        E_log_det_Omega_k[k] = p * np.log(2) + log_det_B_k - digamma_term

        if hat_nu_k[k] > p + 1:
            trace_term[k] = p * hat_nu_k[k] / (hat_lambda_k[k] * (hat_nu_k[k] - p - 1))
        else:
            trace_term[k] = p / hat_lambda_k[k]

    return E_mu_k, E_Omega_k_inv, E_log_det_Omega_k, trace_term

def mahalanobis_matrix(Y, E_mu_k, E_Omega_k_inv, trace_term):
    N, p = Y.shape
    T = E_mu_k.shape[0]
    E_mahalanobis = np.zeros((N, T))
    for k in range(T):
        Y_centered = Y - E_mu_k[k]
        quad = np.sum((Y_centered @ E_Omega_k_inv[k]) * Y_centered, axis=1)
        E_mahalanobis[:, k] = quad + trace_term[k]
    return E_mahalanobis