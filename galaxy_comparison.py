"""
Galaxy data set cluster analysis: comparing CAVI_MFM, truncated DP-CAVI,
and fixed-T finite mixture CAVI.

Data: velocities (km/sec) of 82 galaxies from the Corona Borealis region
(Roeder, 1990; also in R's MASS::galaxies). Used as a benchmark for
Bayesian cluster-number inference in e.g. Richardson & Green (1997) and
Grün, Malsiner-Walli & Frühwirth-Schnatter (2022), "How many data clusters
are in the Galaxy data set? Bayesian cluster analysis in action."
"""

import numpy as np
from scipy.special import psi, gammaln, logsumexp
from scipy.stats import invwishart
import matplotlib.pyplot as plt

# Import your MFM implementation
# (adjust the import path/module name to wherever CAVI_MFM is defined)
from CAVI.mfm import CAVI_MFM
from CAVI.baselines import DPMixtureCAVI, FiniteMixtureCAVI


# ============================================================
# Galaxy data set (n = 82), velocities in km/sec
# ============================================================
GALAXY_VELOCITIES = np.array([
    9172, 9350, 9483, 9558, 9775, 10227, 10406, 16084, 16170, 18419,
    18552, 18600, 18927, 19052, 19070, 19330, 19343, 19349, 19440, 19473,
    19529, 19541, 19547, 19663, 19846, 19856, 19863, 19914, 19918, 19973,
    19989, 20166, 20175, 20179, 20196, 20215, 20221, 20415, 20629, 20795,
    20821, 20846, 20875, 20986, 21137, 21492, 21701, 21814, 21921, 21960,
    22185, 22209, 22242, 22249, 22314, 22374, 22495, 22746, 22747, 22888,
    22914, 23206, 23241, 23263, 23484, 23538, 23542, 23666, 23706, 23711,
    24129, 24285, 24289, 24366, 24717, 24990, 25633, 26960, 26995, 32065,
    32789, 34279
], dtype=float)

# ============================================================
# Infer K for CAVI_MFM by thresholding responsibilities marginalized over q(K)
# ============================================================
def infer_K_mfm(model, threshold_frac=0.02, min_count=2.0):
    N, T = model.N, model.T
    w_nk = np.zeros((N, T))
    for kappa in range(1, T + 1):
        w_nk[:, :kappa] += model.kappa_prob[kappa - 1] * model.r_nk[kappa - 1]
    effective_counts = w_nk.sum(axis=0)
    count_threshold = max(min_count, threshold_frac * N)
    active_mask = effective_counts >= count_threshold
    inferred_K = int(active_mask.sum())
    return inferred_K, effective_counts, active_mask


# ============================================================
# Main: fit all three methods to the Galaxy data set
# ============================================================
if __name__ == '__main__':
    # --- Prepare data ---
    # Standardize velocities to km/sec / 1000 for numerical stability
    # (common convention in the mixture-modeling literature for this data set)
    Y_raw = GALAXY_VELOCITIES / 1000.0
    Y = Y_raw.reshape(-1, 1)   # N x p, p = 1
    N, p = Y.shape
    print(f"Galaxy data set: N = {N} galaxies, p = {p} (velocity in 1000 km/sec)")
    print(f"Range: [{Y.min():.2f}, {Y.max():.2f}], mean = {Y.mean():.2f}, std = {Y.std():.2f}")

    # --- Recommended settings, approximating Grün, Malsiner-Walli &
    #     Frühwirth-Schnatter (2022)'s "sparse solution" recipe for the Galaxy data ---
    T_truncation = 20   # generous; their U(1,30) shows 30 is never binding

    # Prior on K: shifted BNB(1,4,3), exactly as in the paper
    ALPHA_LAMBDA_PRIOR = 1
    ALPHA_PI_PRIOR = 4
    BETA_PI_PRIOR = 3

    # Dynamic MFM with small alpha_eta -> sparse K_+ (paper's recommended regime)
    # E[alpha_eta] = A0 / B0_GAMMA ~= 0.01
    A0, B0_GAMMA = 1, 0.3

    # Component prior: approximate independence-prior recommendation
    # (b0=data midpoint, "large" C0/B0 for coarse, sparsity-inducing shape)
    LAMBDA0 = 0.01            # diffuse prior on mu_k (mimics large B0=630)
    NU0 = 2.0                 # = 2 * c0, c0 = 2 (paper's fixed value)
    B0_MU = np.array([21.73]) # midpoint of (9.172, 34.279)
    B0_SCALE = np.array([[2.0]])  # = 2 * C0, C0 = 12.5 (paper's sparse-inducing value)

    OCCUPANCY_THRESHOLD_FRAC = 0.02   # ~2% of N
    OCCUPANCY_THRESHOLD_MIN = 2.0

    # --- Fit 1: CAVI_MFM ---
    print("\n--- Fitting CAVI_MFM ---")
    mfm = CAVI_MFM(Y, T=T_truncation,
                    a0=A0, b0=B0_GAMMA,
                    lambda0=LAMBDA0, nu0=NU0, b0_mu=B0_MU, B0=B0_SCALE)
    mfm.fit(max_iter=200, tol=1e-5, verbose=False)
    k_mfm, counts_mfm, active_mfm = infer_K_mfm(mfm, OCCUPANCY_THRESHOLD_FRAC, OCCUPANCY_THRESHOLD_MIN)
    map_K_mfm = int(np.argmax(mfm.kappa_prob)) + 1

    print(f"MFM: threshold-based K = {k_mfm},  MAP(q(K)) = {map_K_mfm}")
    print(f"MFM posterior over K: {np.round(mfm.kappa_prob, 3)}")
    print(f"MFM component occupancies (sorted): {np.round(np.sort(counts_mfm)[::-1], 2)}")

    # --- Fit 2: DP-CAVI ---
    print("\n--- Fitting DP-CAVI ---")
    dp = DPMixtureCAVI(Y, T=T_truncation, s1=A0, s2=B0_GAMMA,
                        lambda0=LAMBDA0, nu0=NU0, b0_mu=B0_MU, B0=B0_SCALE)
    dp.fit(max_iter=200, tol=1e-5, verbose=False)
    k_dp, counts_dp = dp.infer_K(OCCUPANCY_THRESHOLD_FRAC, OCCUPANCY_THRESHOLD_MIN)
    print(f"DP-CAVI: threshold-based K = {k_dp}")
    print(f"DP-CAVI component occupancies (sorted): {np.round(np.sort(counts_dp)[::-1], 2)}")

    # --- Fit 3: Finite-T CAVI ---
    print("\n--- Fitting Finite-T CAVI ---")
    fin = FiniteMixtureCAVI(Y, T=T_truncation, alpha0=A0/B0_GAMMA,
                             lambda0=LAMBDA0, nu0=NU0, b0_mu=B0_MU, B0=B0_SCALE)
    fin.fit(max_iter=200, tol=1e-5, verbose=False)
    k_fin, counts_fin = fin.infer_K(OCCUPANCY_THRESHOLD_FRAC, OCCUPANCY_THRESHOLD_MIN)
    print(f"Finite-T CAVI: threshold-based K = {k_fin}")
    print(f"Finite-T CAVI component occupancies (sorted): {np.round(np.sort(counts_fin)[::-1], 2)}")

    # --- Summary comparison ---
    print("\n" + "=" * 60)
    print("GALAXY DATA SET: COMPARISON OF INFERRED K")
    print("=" * 60)
    print(f"{'Method':<20}{'Inferred K':<15}")
    print(f"{'CAVI_MFM (yours)':<20}{k_mfm:<15}")
    print(f"{'DP-CAVI':<20}{k_dp:<15}")
    print(f"{'Finite-T CAVI':<20}{k_fin:<15}")
    print("\nFor reference, published estimates for this data set commonly range")
    print("from 3 to 9 clusters depending on model/prior (see e.g. Richardson &")
    print("Green 1997; Grün, Malsiner-Walli & Frühwirth-Schnatter 2022).")

    # --- Visualization: histogram with fitted component means ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    def plot_fit(ax, model_name, hat_b_k, counts, threshold, k_inferred, color):
        ax.hist(Y_raw, bins=30, density=True, alpha=0.35, color="gray", edgecolor="white")
        active = counts >= threshold
        for k in np.where(active)[0]:
            ax.axvline(hat_b_k[k, 0] * 1000, color=color, linewidth=2)
        ax.set_title(f"{model_name}: inferred K = {k_inferred}")
        ax.set_ylabel("density")

    threshold_val = max(OCCUPANCY_THRESHOLD_MIN, OCCUPANCY_THRESHOLD_FRAC * N)
    plot_fit(axes[0], "CAVI_MFM", mfm.hat_b_k, counts_mfm, threshold_val, k_mfm, "crimson")
    plot_fit(axes[1], "DP-CAVI", dp.hat_b_k, counts_dp, threshold_val, k_dp, "royalblue")
    plot_fit(axes[2], "Finite-T CAVI", fin.hat_b_k, counts_fin, threshold_val, k_fin, "darkgreen")

    axes[-1].set_xlabel("velocity (km/sec)")
    plt.tight_layout()
    plt.savefig("galaxy_data_comparison.png", dpi=150)
    plt.show()