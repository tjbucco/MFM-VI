"""
param_recovery.py

Utilities for assessing recovery of mixture-model component parameters
(means, covariances, mixing weights) across methods that suffer from
label switching (the component index order is arbitrary and generally
different from the generative order, and different between methods).

Two complementary assessments are provided:

1. `match_and_score_params`: matches estimated components to true
   components via the Hungarian algorithm (minimizing total Euclidean
   distance between means), then reports per-component errors in mu,
   Omega, and eta for the matched pairs. This only makes sense when
   inferred K and true K_plus are reasonably close; components are
   matched greedily/optimally on a rectangular cost matrix so extra or
   missing components are simply left unmatched and reported separately.

2. `clustering_agreement`: Adjusted Rand Index and Normalized Mutual
   Information between the true hard labels and each method's inferred
   hard labels (argmax responsibility). These are invariant to label
   permutation by construction, so they're a useful cross-check that
   doesn't depend on getting the matching in (1) right.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------------------------------------------------------------
# 1) Extracting point estimates from a fitted model
# ---------------------------------------------------------------------
def get_posterior_estimates(model, method, min_eta_threshold=1e-3):
    """
    Extract point estimates of (mu_k, Omega_k, eta_k) and hard cluster assignments
    from fitted MFM / CAVI model objects using NormalWishartAtoms.
    """
    p_dim = getattr(model, "p", getattr(model, "D", None))

    # -----------------------------------------------------------------
    # 1. Force expectation computation if using NormalWishartAtoms
    # -----------------------------------------------------------------
    if hasattr(model, "_compute_atom_expectations"):
        model._compute_atom_expectations()

    # -----------------------------------------------------------------
    # 2. Extract Component Means (mus_est_full)
    # -----------------------------------------------------------------
    if hasattr(model, "component_means"):
        mus_est_full = np.atleast_2d(model.component_means())
    elif hasattr(model, "m_hat"):
        mus_est_full = np.atleast_2d(model.m_hat)
    else:
        raise AttributeError(f"[{method}] Could not find mean attributes (m_hat or component_means).")

    # -----------------------------------------------------------------
    # 3. Extract Component Covariances / Precision (Omegas_est_full)
    # -----------------------------------------------------------------
    if hasattr(model, "component_covariances"):
        covs = np.atleast_3d(model.component_covariances())
        Omegas_est_full = covs  # Note: Use np.linalg.inv(covs) if your metric expects Precision
    elif hasattr(model, "C_hat") and hasattr(model, "nu_hat"):
        nu_hat = np.asarray(model.nu_hat)
        denom = np.maximum(nu_hat - p_dim - 1.0, 1e-6)
        covs = np.atleast_3d(model.C_hat / denom[:, None, None])
        Omegas_est_full = covs
    else:
        raise AttributeError(f"[{method}] Could not find covariance/precision parameters (C_hat or component_covariances).")

    # Guard against 1D / 2D dimensional collapse
    if mus_est_full.ndim == 1:
        mus_est_full = mus_est_full[:, None]
    if Omegas_est_full.ndim == 2:
        Omegas_est_full = Omegas_est_full[None, :, :]

    # -----------------------------------------------------------------
    # 4. Extract Responsibilities across dynamic (MFM) and fixed models
    # -----------------------------------------------------------------
    if hasattr(model, "r") and isinstance(model.r, list):  # Untied MFM Structure
        N, T = model.N, model.T
        w_nk = np.zeros((N, T))
        
        for kappa in range(1, T + 1):
            rk = model.r[kappa - 1]
            if rk is not None:
                w_nk[:, :kappa] += model.pi[kappa - 1] * rk
            elif model.pi[kappa - 1] > 0 and hasattr(model, "_expected_log_lik"):
                E_log_lik = model._expected_log_lik(model.Y)
                rk = model._resp_for(E_log_lik, kappa)
                w_nk[:, :kappa] += model.pi[kappa - 1] * rk

    elif hasattr(model, "r") and model.r is not None:  # Standard Single Array (DP / Finite)
        w_nk = np.asarray(model.r)
    elif hasattr(model, "mixture_weights"):
        eta_est_full = np.asarray(model.mixture_weights())
        w_nk = np.tile(eta_est_full, (model.N, 1))
    else:
        raise AttributeError(f"[{method}] Could not find responsibility attribute 'r'.")

    # -----------------------------------------------------------------
    # 5. Compute Empirical Mixture Weights & Filter Active Atoms
    # -----------------------------------------------------------------
    hard_labels_full = np.argmax(w_nk, axis=1)
    
    effective_counts = w_nk.sum(axis=0)
    if effective_counts.sum() > 0:
        eta_est_full = effective_counts / effective_counts.sum()
    elif hasattr(model, "mixture_weights"):
        eta_est_full = np.asarray(model.mixture_weights())

    # Retain components above mass threshold
    active = np.where(eta_est_full >= min_eta_threshold)[0]
    
    if len(active) == 0:
        active = np.unique(hard_labels_full)

    # Remap hard cluster labels to range [0, len(active)-1]
    idx_map = {old: new for new, old in enumerate(active)}
    hard_labels = np.array([idx_map.get(l, -1) for l in hard_labels_full])

    # Reassign orphaned points to nearest active cluster
    unassigned = hard_labels == -1
    if np.any(unassigned):
        hard_labels[unassigned] = np.argmax(w_nk[unassigned][:, active], axis=1)

    mus_est = mus_est_full[active]
    Omegas_est = Omegas_est_full[active]
    eta_est = eta_est_full[active]
    eta_est = eta_est / eta_est.sum()

    return mus_est, Omegas_est, eta_est, hard_labels
# ---------------------------------------------------------------------
# 2) Matching estimated components to true components
# ---------------------------------------------------------------------
def match_and_score_params(true_mus, true_Omegas, true_eta,
                            est_mus, est_Omegas, est_eta):
    """
    Match estimated components to true components (Hungarian algorithm on
    Euclidean distance between means), then compute errors for the matched
    pairs. Unmatched true/estimated components (when K_true != K_est) are
    reported as counts, not scored.

    Returns a dict with:
        n_matched, n_missing (true comps with no match),
        n_spurious (est comps with no match),
        mean_mu_error   (avg Euclidean distance ||mu_true - mu_est||),
        mean_omega_error (avg Frobenius norm ||Omega_true - Omega_est||_F),
        mean_eta_error  (avg abs difference in mixing weight),
        matching        (list of (true_idx, est_idx) pairs)
    """
    K_true, K_est = true_mus.shape[0], est_mus.shape[0]

    # cost[i, j] = distance between true component i and estimated component j
    cost = np.linalg.norm(
        true_mus[:, None, :] - est_mus[None, :, :], axis=-1
    )
    row_idx, col_idx = linear_sum_assignment(cost)

    mu_errors, omega_errors, eta_errors = [], [], []
    matching = []
    for i, j in zip(row_idx, col_idx):
        mu_errors.append(np.linalg.norm(true_mus[i] - est_mus[j]))
        omega_errors.append(np.linalg.norm(true_Omegas[i] - est_Omegas[j], ord="fro"))
        eta_errors.append(abs(true_eta[i] - est_eta[j]))
        matching.append((int(i), int(j)))

    n_matched = len(row_idx)
    return {
        "n_matched": n_matched,
        "n_missing": K_true - n_matched,
        "n_spurious": K_est - n_matched,
        "mean_mu_error": float(np.mean(mu_errors)) if mu_errors else np.nan,
        "mean_omega_error": float(np.mean(omega_errors)) if omega_errors else np.nan,
        "mean_eta_error": float(np.mean(eta_errors)) if eta_errors else np.nan,
        "matching": matching,
    }


# ---------------------------------------------------------------------
# 3) Label-permutation-invariant clustering agreement
# ---------------------------------------------------------------------
def clustering_agreement(true_labels, est_labels):
    """
    ARI and NMI between true and estimated hard cluster assignments.
    Both are invariant to relabeling, so they don't require the matching
    step above and serve as an independent sanity check.
    """
    return {
        "ari": adjusted_rand_score(true_labels, est_labels),
        "nmi": normalized_mutual_info_score(true_labels, est_labels),
    }


# ---------------------------------------------------------------------
# 4) Convenience wrapper: run both assessments for one fitted model
# ---------------------------------------------------------------------
def assess_parameter_recovery(model, method, true_mus, true_Omegas, true_eta, true_labels):
    """
    Full parameter-recovery assessment for one fitted model against the
    known ground truth for a trial. Returns a flat dict suitable for
    appending to a results list alongside the existing K-inference metrics.
    """
    est_mus, est_Omegas, est_eta, hard_labels = get_posterior_estimates(model, method)

    param_scores = match_and_score_params(
        true_mus, true_Omegas, true_eta, est_mus, est_Omegas, est_eta
    )
    cluster_scores = clustering_agreement(true_labels, hard_labels)

    out = {}
    out.update({f"{method}_{k}": v for k, v in param_scores.items() if k != "matching"})
    out.update({f"{method}_{k}": v for k, v in cluster_scores.items()})
    return out