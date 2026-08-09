import numpy as np
from scipy.stats import invwishart, betanbinom as BNB

from CAVI.mfm import CAVI_MFM
from CAVI.baselines import DPMixtureCAVI, FiniteMixtureCAVI


# Global Hyperparameters
p = 5
T_truncation = 15

ALPHA_LAMBDA_PRIOR = 1      # BNB prior on K (must match CAVI_MFM internals)
ALPHA_PI_PRIOR = 4
BETA_PI_PRIOR = 30

A0, B0_GAMMA = 20.0, 1.030     # Gamma(shape=a0, rate=b0) prior on alpha_eta

# NIW prior
LAMBDA0 = 0.0005
NU0 = p + 2.0
B0_MU = np.zeros(p)
B0_SCALE = np.eye(p) * (NU0 - p - 1)         # so E[Omega_k] = I under the prior

N_TRIALS = 100
N_PER_TRIAL = 500

OCCUPANCY_THRESHOLD_FRAC = 0.0001
OCCUPANCY_THRESHOLD_MIN = 2.0


def deterministic_well_separated_means(K, p, radius=9.0, center=None):
    """
    Places K means deterministically on a circle of given radius, evenly
    spaced by angle, embedded in the first two coordinates of R^p (extra
    dimensions, if any, are set to `center`'s value, default 0).

    Guarantees exact pairwise distance between adjacent means:
        min_sep = 2 * radius * sin(pi / K)
    with no randomness and no rejection sampling.
    """
    if center is None:
        center = np.zeros(p)
    means = np.tile(center.astype(float), (K, 1))  # K x p, defaults to center

    angles = 2 * np.pi * np.arange(K) / K
    means[:, 0] += radius * np.cos(angles)
    if p >= 2:
        means[:, 1] += radius * np.sin(angles)
    # if p == 1, only the first coordinate varies (means will coincide in
    # pairs for K > 2 since a 1-D circle embedding is degenerate; use p>=2)

    return means

# --- Step 1: K ~ BNB(alpha_lambda, alpha_pi, beta_pi), truncated to {1,...,T} ---
def sample_true_K(T, rng, a_lambda=ALPHA_LAMBDA_PRIOR, a_pi=ALPHA_PI_PRIOR, b_pi=BETA_PI_PRIOR):
    probs = np.array([BNB.pmf(t, a_lambda, a_pi, b_pi) for t in range(T)])
    probs[0] = 0
    probs /= probs.sum()
    t = rng.choice(T, p=probs)
    return t + 1  # support {0,...,T-1} -> K in {1,...,T}

# --- Steps 2-5: ancestral sample of (alpha_eta, eta, {mu_k,Omega_k}, S, Y) given K ---
def generate_data_from_model(K, p, N, rng,
                                a0=A0, b0=B0_GAMMA,
                                lambda0=LAMBDA0, nu0=NU0, b0_mu=B0_MU, B0=B0_SCALE,
                                min_sep=6.0, mean_radius=9.0):
    # alpha_eta ~ Gamma(shape=a0, rate=b0)  => numpy scale = 1/rate
    alpha_eta = rng.gamma(shape=a0, scale=1.0 / b0)

    # eta | alpha_eta, K ~ Dirichlet(alpha_eta/K, ..., alpha_eta/K)
    eta = rng.dirichlet(np.full(K, alpha_eta / K))

    # (mu_k, Omega_k) ~ NIW(b0_mu, lambda0, B0, nu0) iid for k=1..K
    mus = np.zeros((K, p))

    #mus = deterministic_well_separated_means(K, p, radius=radius_k)
    Omegas = np.zeros((K, p, p))
    for k in range(K):
        # Omega_k ~ Inverse-Wishart(scale=B0, df=nu0)  [covariance matrix, per E_Omega_k_inv formula]
        Omega_k = invwishart.rvs(df=nu0, scale=B0, random_state=rng)
        if p == 1:
            Omega_k = np.array([[Omega_k]])
        # mu_k | Omega_k ~ N(b0_mu, Omega_k / lambda0)
        mu_k = rng.multivariate_normal(b0_mu, Omega_k / lambda0)
        mus[k] = mu_k
        Omegas[k] = Omega_k

    # S_n | eta ~ Categorical(eta),  Y_n | S_n=k ~ N(mu_k, Omega_k)
    labels = rng.choice(K, size=N, p=eta)
    Y = np.zeros((N, p))
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        Y[idx] = rng.multivariate_normal(mus[k], Omegas[k], size=len(idx))
    occupied_K = len(np.unique(labels))

    return Y, labels, mus, Omegas, eta, alpha_eta, occupied_K

# --- Infer K by thresholding responsibilities, marginalized over q(K) [MFM only] ---
def infer_K_by_responsibility_threshold(model, threshold_frac=OCCUPANCY_THRESHOLD_FRAC,
                                            min_count=OCCUPANCY_THRESHOLD_MIN):
    N, T = model.N, model.T
    w_nk = np.zeros((N, T))
    for kappa in range(1, T + 1):
        w_nk[:, :kappa] += model.kappa_prob[kappa - 1] * model.r_nk[kappa - 1]
    effective_counts = w_nk.sum(axis=0)
    count_threshold = max(min_count, threshold_frac * N)
    active_mask = effective_counts >= count_threshold
    inferred_K = int(active_mask.sum())
    return inferred_K, effective_counts, active_mask

def radius_for_target_separation(K, target_sep):
    """Radius needed so adjacent means are exactly `target_sep` apart."""
    return target_sep / (2 * np.sin(np.pi / K))


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_component_means(means, radius=None, Y=None, labels=None, title="Component Means"):
    """
    Visualize component means in 2D (or the first two dimensions of R^p).
    """
    K = means.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))

    if Y is not None:
        if labels is not None:
            cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
            for k in range(K):
                idx = labels == k
                ax.scatter(Y[idx, 0], Y[idx, 1], s=12, alpha=0.4,
                        color=cmap(k % cmap.N), label=f"cluster {k}")
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=12, alpha=0.3, color="gray")

    if radius is not None:
        circle = patches.Circle((0, 0), radius, fill=False,
                                linestyle="--", color="lightgray", linewidth=1)
        ax.add_patch(circle)

    ax.scatter(means[:, 0], means[:, 1], s=200, marker="X",
            color="black", edgecolor="white", linewidth=1.5, zorder=5,
            label="component means")
    for k, (mx, my) in enumerate(means[:, :2]):
        ax.annotate(f"K={k}", (mx, my), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title)
    if not (Y is not None and labels is None):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    rng = np.random.default_rng(2024)

    # --- Run repeated trials: fit MFM, DP-CAVI, and Finite-T CAVI on the SAME data ---
    results = []
    plot_on = False
    show_confusion=False
    show_pertrialcomparison=False
    per_trial_statistics=False
    
    for trial in range(N_TRIALS):
        true_K = sample_true_K(T_truncation, rng)
        radius_k = radius_for_target_separation(true_K, target_sep=6.0)

        Y, labels, true_mus, true_Omegas, true_eta, true_alpha_eta, true_occupied_K = generate_data_from_model(
            true_K, p, N_PER_TRIAL, rng, mean_radius=radius_k
        )

        if trial < 3 and plot_on == True:
            plot_component_means(true_mus, Y=Y, labels=labels,
                       title=f"Trial {trial+1}: True K_plus={true_occupied_K}")

        if per_trial_statistics == True:
            print(f"\n=== Trial {trial+1}/{N_TRIALS}: true K_plus = {true_occupied_K}, "
                f"alpha_eta = {true_alpha_eta:.2f}, N = {N_PER_TRIAL} ===\nlabels = {np.bincount(labels)}")
        else:
            progress = (trial + 1) / N_TRIALS
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "-" * (bar_length - filled)
            print(f"\rProgress: |{bar}| {trial}/{N_TRIALS} Trials", end="", flush=True)

        # 1) My CAVI_MFM (BNB prior over K, Gamma prior on alpha_eta, Dirichlet weights)
        model = CAVI_MFM(Y, T=T_truncation,
                          #a0=1.0, b0=0.3,
                          a0=A0, b0=B0_GAMMA,
                          lambda0=LAMBDA0, nu0=NU0,
                          #b0_mu=B0_MU, B0=B0_SCALE,
                          alpha_lambda_prior=ALPHA_LAMBDA_PRIOR, alpha_pi_prior=ALPHA_PI_PRIOR, beta_pi_prior=BETA_PI_PRIOR
                          )
        model.fit(max_iter=100, tol=1e-4, verbose=False)
        k_mfm, effective_counts, _ = infer_K_by_responsibility_threshold(model)

        # 2) DP mixture CAVI (stick-breaking, truncated at T)
        dp = DPMixtureCAVI(Y, T=T_truncation, s1=A0, s2=B0_GAMMA,
                            lambda0=LAMBDA0, nu0=NU0, #b0_mu=B0_MU, B0=B0_SCALE
                            )
        dp.fit(max_iter=100, tol=1e-4, verbose=False)
        k_dp, _ = dp.infer_K()

        # 3) Finite mixture CAVI with fixed T components
        fin = FiniteMixtureCAVI(Y, T=T_truncation, alpha0=A0 / B0_GAMMA,
                                 lambda0=LAMBDA0, nu0=NU0, 
                                 #b0_mu=B0_MU, B0=B0_SCALE
                                 )
        fin.fit(max_iter=100, tol=1e-4, verbose=False)
        k_fin, _ = fin.infer_K()

        results.append({
            "trial": trial + 1,
            "true_K": true_K,
            "true_Kplus": true_occupied_K,
            "k_mfm": k_mfm,
            "k_dp": k_dp,
            "k_fin": k_fin,
            "correct_mfm": k_mfm == true_occupied_K,
            "correct_dp": k_dp == true_occupied_K,
            "correct_fin": k_fin == true_occupied_K,
            "abs_error_mfm": abs(k_mfm - true_occupied_K),
            "abs_error_dp": abs(k_dp - true_occupied_K),
            "abs_error_fin": abs(k_fin - true_occupied_K),
        })

        if per_trial_statistics == True:
            sorted_counts = np.round(np.sort(effective_counts)[::-1], 1)
            print(f"  MFM K_hat={k_mfm}   DP K_hat={k_dp}   Finite-T K_hat={k_fin}")
            print(f"  MFM posterior K: {np.round(model.kappa_prob, 3)}")
            print(f"  MFM component occupancies (sorted): {sorted_counts}")

    # --- Summarize accuracy across all trials, per method ---
    true_Ks = np.array([r["true_Kplus"] for r in results])

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: MFM vs. truncated DP-CAVI vs. fixed-T finite CAVI")
    print("=" * 70)
    print(f"Number of trials: {N_TRIALS}")
    print(f"Occupancy threshold: {max(OCCUPANCY_THRESHOLD_MIN, OCCUPANCY_THRESHOLD_FRAC*N_PER_TRIAL):.0f}")

    for method_key, method_name in [("mfm", "MFM (proposed)"), ("dp", "DP-CAVI"), ("fin", "Finite-T CAVI")]:
        K_hats = np.array([r[f"k_{method_key}"] for r in results])
        abs_err = np.array([r[f"abs_error_{method_key}"] for r in results])
        correct = np.array([r[f"correct_{method_key}"] for r in results])

        print(f"\n[{method_name}]")
        print(f"  Exact-match accuracy:  {correct.mean():.3f}")
        print(f"  Within +/-1 accuracy:  {(abs_err <= 1).mean():.3f}")
        print(f"  Mean abs error:        {abs_err.mean():.3f}")
        print(f"  RMSE:                  {np.sqrt((abs_err**2).mean()):.3f}")

        if show_confusion==True:
            max_K_seen = max(true_Ks.max(), K_hats.max())
            confusion = np.zeros((max_K_seen, max_K_seen), dtype=int)
            for tk, kh in zip(true_Ks, K_hats):
                confusion[tk - 1, kh - 1] += 1
            print(f"  Confusion matrix (rows = true K, cols = inferred K):")
            header = "        " + "".join(f"K={j+1:<4}" for j in range(max_K_seen))
            print(header)
            for i in range(max_K_seen):
                row_label = f"  K={i+1:<4}"
                row_vals = "".join(f"{confusion[i, j]:<6}" for j in range(max_K_seen))
                print(f"{row_label}{row_vals}")

    if show_pertrialcomparison==True:
        print("\nPer-trial comparison:")
        print(f"{'Trial':<8}{'True Kplus':<10}{'MFM':<8}{'DP':<8}{'Finite-T':<10}")
        for r in results:
            print(f"{r['trial']:<8}{r['true_Kplus']:<10}{r['k_mfm']:<8}{r['k_dp']:<8}{r['k_fin']:<10}")