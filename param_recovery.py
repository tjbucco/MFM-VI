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
def get_posterior_estimates(model, method):
    """
    Extract point estimates of (mu_k, Omega_k, eta_k) and a hard cluster
    assignment (argmax responsibility per data point) from a fitted CAVI
    model object.

    Confirmed against CAVI_MFM's source:
        - mu_k estimate    : model.hat_b_k                 (T x p)
        - Omega_k estimate : model.hat_B_k / (model.hat_nu_k - p - 1)
                              (matches CAVI_MFM._compute_expectations's E_Omega_k_inv
                               derivation, just un-inverted -- E[Omega_k], not E[Omega_k^-1])
        - responsibilities : model.r_nk for CAVI_MFM -- a LIST of length T
                              indexed by truncation level kappa, needs
                              marginalizing over model.kappa_prob, same as
                              infer_K_by_responsibility_threshold. For
                              DPMixtureCAVI / FiniteMixtureCAVI it's
                              model.phi -- a single N x T array (no q(K),
                              so no marginalization needed).
        - eta_k estimate   : not read from a single clean posterior object
                              (CAVI_MFM's weights live in per-kappa Dirichlet
                              parameters; DP's live in stick-breaking Beta
                              params gamma1/gamma2; Finite's live in
                              alpha_hat). Estimated empirically instead, as
                              the responsibility mass per component
                              (marginalized over q(K) for MFM), normalized --
                              consistent with how effective_counts is
                              computed in the main script, and directly
                              comparable across all three methods.

    Returns
    -------
    mus_est : (K_est, p) array
    Omegas_est : (K_est, p, p) array
    eta_est : (K_est,) array
    hard_labels : (N,) int array, values in [0, K_est)
    """
    p_dim = model.p

    # --- mu_k, Omega_k ---
    if hasattr(model, "hat_b_k") and hasattr(model, "hat_B_k") and hasattr(model, "hat_nu_k"):
        mus_est_full = np.asarray(model.hat_b_k)                      # (T, p)
        hat_B_k = np.asarray(model.hat_B_k)                           # (T, p, p)
        hat_nu_k = np.asarray(model.hat_nu_k)                         # (T,)
        denom = np.clip(hat_nu_k - p_dim - 1, 1e-6, None)
        Omegas_est_full = hat_B_k / denom[:, None, None]              # (T, p, p)
    else:
        raise AttributeError(
            f"[{method}] couldn't find hat_b_k/hat_B_k/hat_nu_k -- "
            "adjust get_posterior_estimates() for this class's attribute names"
        )

    # --- effective (marginalized) responsibility mass per component ---
    if hasattr(model, "r_nk"):
        # CAVI_MFM-style: list over truncation level kappa, needs marginalizing
        # over q(K) exactly as infer_K_by_responsibility_threshold does.
        if not hasattr(model, "kappa_prob"):
            raise AttributeError(
                f"[{method}] r_nk found but no kappa_prob -- "
                "can't marginalize over q(K)"
            )
        N, T = model.N, model.T
        w_nk = np.zeros((N, T))
        for kappa in range(1, T + 1):
            w_nk[:, :kappa] += model.kappa_prob[kappa - 1] * model.r_nk[kappa - 1]
    elif hasattr(model, "phi"):
        # DPMixtureCAVI / FiniteMixtureCAVI: single N x T responsibility
        # matrix, no kappa marginalization (no model selection over K).
        w_nk = np.asarray(model.phi)
    else:
        raise AttributeError(
            f"[{method}] couldn't find a responsibility attribute "
            "(tried r_nk, phi) -- adjust get_posterior_estimates()"
        )

    hard_labels = np.argmax(w_nk, axis=1)
    effective_counts = w_nk.sum(axis=0)                # (T,)
    eta_est_full = effective_counts / effective_counts.sum()

    # keep only components that actually have mass (avoids matching against
    # empty/unused truncation slots), then re-index labels to match
    active = np.unique(hard_labels)
    idx_map = {old: new for new, old in enumerate(active)}
    hard_labels = np.array([idx_map[l] for l in hard_labels])

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