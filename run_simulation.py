import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

import numpy as np
from scipy.stats import invwishart, betanbinom as BNB

from CAVI.mfm import CAVI_MFM_Gaussian
from CAVI.baselines import DPMixtureCAVI_Gaussian, FiniteMixtureCAVI_Gaussian
from param_recovery import assess_parameter_recovery

# =====================================================================
# Global Hyperparameters
# =====================================================================
p = 2
T_truncation = 10

ALPHA_LAMBDA_PRIOR = 1
ALPHA_PI_PRIOR = 4
BETA_PI_PRIOR = 3

A0, B0_GAMMA = 5.0, 0.10
#A0_true, B0_GAMMA_true = 1.0, 0.30
force_equal_sizes_flag = False

# NIW prior
LAMBDA0 = 0.05
NU0 = p + max(4, p // 2)            # e.g., 15 for p=10
B0_MU = np.zeros(p)
B0_SCALE = np.eye(p) * 1 * (NU0 - p - 1)

N_TRIALS = 100
N_PER_COMP = max(12, 1 * p)  # e.g., 1200 for p=10

# Thresholding parameters (legacy, kept for comparison)
OCCUPANCY_THRESHOLD_FRAC = 0.0001
OCCUPANCY_THRESHOLD_MIN = 2.0

# Monte Carlo samples for posterior_summaries
N_MC_SAMPLES = 5000
N_RESTARTS = 1


# =====================================================================
# Data generation utilities
# =====================================================================
def deterministic_well_separated_means(K, p, radius=9.0, center=None):
    if center is None:
        center = np.zeros(p)
    means = np.tile(center.astype(float), (K, 1))
    angles = 2 * np.pi * np.arange(K) / K
    means[:, 0] += radius * np.cos(angles)
    if p >= 2:
        means[:, 1] += radius * np.sin(angles)
    return means


def sample_true_K(T, rng, a_lambda=ALPHA_LAMBDA_PRIOR, a_pi=ALPHA_PI_PRIOR,
                  b_pi=BETA_PI_PRIOR):
    probs = np.array([BNB.pmf(t, a_lambda, a_pi, b_pi) for t in range(T)])
    probs[0] = 0
    probs /= probs.sum()
    t = rng.choice(T, p=probs)
    return t + 1


def generate_data_from_model(K, p, N, rng,
                             a0=A0, b0=B0_GAMMA,
                             beta0=LAMBDA0, nu0=NU0, b0_mu=B0_MU, B0=B0_SCALE,
                             min_sep=6.0, mean_radius=9.0, force_equal_sizes=False):
    alpha_eta = rng.gamma(shape=a0, scale=1.0 / b0)
    eta = rng.dirichlet(np.full(K, alpha_eta / K))
    eta = np.full(K, 1 / K)
    mus = np.zeros((K, p))
    Omegas = np.zeros((K, p, p))
    for k in range(K):
        Omega_k = invwishart.rvs(df=nu0, scale=B0, random_state=rng)
        if p == 1:
            Omega_k = np.array([[Omega_k]])
        mu_k = rng.multivariate_normal(b0_mu, Omega_k / beta0)
        mus[k] = mu_k
        Omegas[k] = Omega_k
        
    TRUE_MU = np.array([
    [-1.1875, 0.0625],  # pair A, large
    [ 0.6875, 0.4375],  # pair A, medium
    [-0.2500, 3.3750],  # pair B, medium
    [ 1.2500, 2.6250],  # pair B, SMALL
    ])
    # TRUE_MU = np.array(
    #     [
    #         [-1.0234375, 0.1093750],  # pair A, large
    #         [0.5234375, 0.3906250],  # pair A, medium
    #         [-0.0625000, 3.2812500],  # pair B, medium
    #         [1.0625000, 2.7187500],  # pair B, SMALL
    #     ]
    # )
    TRUE_COV = [
    np.array([[1.0, 0.3], [0.3, 0.5]]),  # tilted, moderate
    np.array([[0.5, -0.1], [-0.1, 0.7]]),  # tilted other way
    np.array([[0.7, 0.0], [0.0, 0.3]]),  # horizontal ellipse
    np.array([[0.25, 0.05], [0.05, 0.30]]),  # small, tight
    ]
    
    for k in range(K):
            Omega_k = TRUE_COV[k]
            if p == 1:
                Omega_k = np.array([[Omega_k]])
            mu_k = TRUE_MU[k]
            mus[k] = mu_k
            Omegas[k] = Omega_k

    labels = rng.choice(K, size=N, p=eta)
    if force_equal_sizes:
        labels = np.repeat(np.arange(K), N // K)
        remainder = N % K
        if remainder > 0:
            extra_labels = rng.choice(K, size=remainder, replace=False)
            labels = np.concatenate([labels, extra_labels])
        rng.shuffle(labels)

    Y = np.zeros((N, p))
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        Y[idx] = rng.multivariate_normal(mus[k], Omegas[k], size=len(idx))
    occupied_K = len(np.unique(labels))
    return Y, labels, mus, Omegas, eta, alpha_eta, occupied_K


def radius_for_target_separation(K, target_sep):
    return target_sep / (2 * np.sin(np.pi / K))


# =====================================================================
# K_+ inference helpers
# =====================================================================
def infer_K_by_responsibility_threshold(model, threshold_frac=OCCUPANCY_THRESHOLD_FRAC,
                                        min_count=OCCUPANCY_THRESHOLD_MIN):
    """Legacy: threshold on marginal effective counts for the MFM."""
    N, T = model.N, model.T
    w_nk = np.zeros((N, T))
    for kappa in range(1, T + 1):
        if model.r[kappa - 1] is None:
            continue
        w_nk[:, :kappa] += model.pi[kappa - 1] * model.r[kappa - 1]
    effective_counts = w_nk.sum(axis=0)
    count_threshold = max(min_count, threshold_frac * N)
    active_mask = effective_counts >= count_threshold
    inferred_K = int(active_mask.sum())
    return inferred_K, effective_counts, active_mask


def extract_posterior_summaries(model, model_key, n_samples=N_MC_SAMPLES, seed=None):
    """
    Call posterior_summaries() on any of the three models and return a
    flat dict of results keyed by model_key, suitable for merging into
    the per-trial results dict.

    For the MFM, which has q(K), we also extract K_map and E_q[K].
    For the baselines (DP, Finite-T), those fields are set to None.
    """
    ps = model.posterior_summaries(n_samples=n_samples, seed=seed)

    E_Kplus = ps['E_K_plus']
    K_plus_pmf = ps['K_plus_pmf']
    K_plus_round = int(round(E_Kplus))

    # Compute mode from pmf (robust: works whether or not the model
    # already includes 'K_plus_mode' in its output)
    K_plus_mode = ps.get('K_plus_mode', int(np.argmax(K_plus_pmf)) + 1)

    out = {
        f"{model_key}_E_Kplus":          E_Kplus,
        f"{model_key}_Kplus_mode":       K_plus_mode,
        f"{model_key}_Kplus_round":      K_plus_round,
        f"{model_key}_Kplus_pmf":        K_plus_pmf,
        f"{model_key}_assignments_ps":   ps['assignments'],
    }

    # MFM-only fields
    if model_key == "mfm":
        out[f"{model_key}_K_map"]  = ps.get('K_map', None)
        out[f"{model_key}_E_K"]    = ps.get('E_K', None)
        out[f"{model_key}_q_K"]    = ps.get('q_K', None)

    return out

# =====================================================================
# Plotting (optional)
# =====================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_component_means(means, radius=None, Y=None, labels=None,
                         title="Component Means"):
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
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_title(title)
    if not (Y is not None and labels is None):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()


# =====================================================================
# Main simulation loop
# =====================================================================
if __name__ == '__main__':
    rng = np.random.default_rng(2024)

    results = []
    plot_on = False
    show_confusion = False
    show_pertrialcomparison = False
    per_trial_statistics = False

    for trial in range(N_TRIALS):
        true_K = sample_true_K(T_truncation, rng)
        #print(f"true_K = {true_K}")
        true_K = 4
        radius_k = radius_for_target_separation(true_K, target_sep=6.0)
        N_PER_TRIAL = N_PER_COMP * true_K
        Y, labels, true_mus, true_Omegas, true_eta, true_alpha_eta, true_occupied_K = \
            generate_data_from_model(
                true_K, p, N_PER_TRIAL, rng, mean_radius=radius_k,
                force_equal_sizes=force_equal_sizes_flag
            )

        if trial < 3 and plot_on:
            plot_component_means(true_mus, Y=Y, labels=labels,
                                title=f"Trial {trial+1}: True K_plus={true_occupied_K}")

        if per_trial_statistics:
            print(f"\n=== Trial {trial+1}/{N_TRIALS}: true K_plus = {true_occupied_K}, "
                  f"alpha_eta = {true_alpha_eta:.2f}, N = {N_PER_TRIAL} ==="
                  f"\nlabels = {np.bincount(labels)}")
        else:
            progress = (trial + 1) / N_TRIALS
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "-" * (bar_length - filled)
            print(f"\rProgress: |{bar}| {trial+1}/{N_TRIALS} Trials",
                  end="", flush=True)

        # ============================================================
        # 1) MFM (proposed)
        # ============================================================
        model = CAVI_MFM_Gaussian(
            Y, T=T_truncation,
            a0=A0, b0=B0_GAMMA,
            beta0=LAMBDA0, nu0=NU0, B0_mode='cavi_mfm',
            alpha_lambda_prior=ALPHA_LAMBDA_PRIOR,
            alpha_pi_prior=ALPHA_PI_PRIOR,
            beta_pi_prior=BETA_PI_PRIOR, init='kmeans_cavi', random_state=42
        )
        model.fit(max_iter=500, tol=1e-4, n_restarts=N_RESTARTS, verbose=False)

        # Legacy thresholding (for backward compat)
        k_mfm_thresh, effective_counts, _ = infer_K_by_responsibility_threshold(model)

        # Posterior summaries (Section 3.6)
        mfm_ps = extract_posterior_summaries(model, "mfm", seed=trial)

        # ============================================================
        # 2) DP mixture CAVI
        # ============================================================
        dp = DPMixtureCAVI_Gaussian(
            Y, T=T_truncation, s1=A0, s2=B0_GAMMA, B0_mode='cavi_mfm',
            beta0=LAMBDA0, nu0=NU0, init='kmeans_cavi'
        )
        dp.fit(max_iter=500, tol=1e-4, n_restarts=N_RESTARTS, verbose=False)

        # Legacy thresholding
        k_dp_thresh, _ = dp.infer_K()

        # Posterior summaries
        dp_ps = extract_posterior_summaries(dp, "dp", seed=trial)

        # ============================================================
        # 3) Finite mixture CAVI
        # ============================================================
        fin = FiniteMixtureCAVI_Gaussian(
            Y, T=T_truncation, alpha0=A0 / B0_GAMMA, B0_mode='cavi_mfm',
            beta0=LAMBDA0, nu0=NU0, init='kmeans_cavi'
        )
        fin.fit(max_iter=500, tol=1e-4, n_restarts=N_RESTARTS, verbose=False)

        # Legacy thresholding
        k_fin_thresh, _ = fin.infer_K()

        # Posterior summaries
        fin_ps = extract_posterior_summaries(fin, "fin", seed=trial)

        # ============================================================
        # Parameter recovery
        # ============================================================
        param_results = {}
        for m, key in [(model, "mfm"), (dp, "dp"), (fin, "fin")]:
            try:
                param_results.update(
                    assess_parameter_recovery(
                        m, key,
                        true_mus=true_mus, true_Omegas=true_Omegas,
                        true_eta=true_eta, true_labels=labels,
                    )
                )
            except AttributeError as e:
                if trial == 0:
                    print(f"\n[param_recovery warning] {e}")

        # ============================================================
        # Assemble results dict
        # ============================================================
        row = {
            "trial": trial + 1,
            "true_K": true_K,
            "true_Kplus": true_occupied_K,
        }

        # --- Legacy thresholding results ---
        row["k_mfm_thresh"] = k_mfm_thresh
        row["k_dp_thresh"]  = k_dp_thresh
        row["k_fin_thresh"] = k_fin_thresh

        # --- Posterior-summaries results for all three methods ---
        row.update(mfm_ps)
        row.update(dp_ps)
        row.update(fin_ps)

        # --- Derived comparison columns ---
        # For each model and each K_+ estimator, compute correctness / error
        for mk in ["mfm", "dp", "fin"]:
            for est_key, est_field in [
                ("thresh",     f"k_{mk}_thresh"),
                ("Kplus_mode", f"{mk}_Kplus_mode"),
                ("Kplus_round", f"{mk}_Kplus_round"),
            ]:
                full_key = f"{mk}_{est_key}"
                k_hat = row[est_field]
                row[f"correct_{full_key}"]   = k_hat == true_occupied_K
                row[f"abs_error_{full_key}"] = abs(k_hat - true_occupied_K)

        # MFM-only: MAP of q(K)
        k_mfm_Kmap = mfm_ps["mfm_K_map"]
        row["correct_mfm_Kmap"]   = k_mfm_Kmap == true_occupied_K
        row["abs_error_mfm_Kmap"] = abs(k_mfm_Kmap - true_occupied_K)

        row.update(param_results)
        results.append(row)

        # --- Per-trial diagnostics ---
        if per_trial_statistics:
            sorted_counts = np.round(np.sort(effective_counts)[::-1], 1)
            print(f"  MFM:  thresh={k_mfm_thresh}  K_map={k_mfm_Kmap}  "
                  f"mode(K+)={mfm_ps['mfm_Kplus_mode']}  "
                  f"E[K+]={mfm_ps['mfm_E_Kplus']:.2f}")
            print(f"  DP:   thresh={k_dp_thresh}  "
                  f"mode(K+)={dp_ps['dp_Kplus_mode']}  "
                  f"E[K+]={dp_ps['dp_E_Kplus']:.2f}")
            print(f"  Fin:  thresh={k_fin_thresh}  "
                  f"mode(K+)={fin_ps['fin_Kplus_mode']}  "
                  f"E[K+]={fin_ps['fin_E_Kplus']:.2f}")
            print(f"  MFM q(K): {np.round(mfm_ps['mfm_q_K'], 3)}")
            print(f"  MFM effective counts (sorted): {sorted_counts}")

    # =====================================================================
    # Summary
    # =====================================================================
    true_Ks = np.array([r["true_Kplus"] for r in results])

    print("\n\n" + "=" * 85)
    print("COMPARISON SUMMARY")
    print("=" * 85)
    print(f"Trials: {N_TRIALS}  |  N: {N_PER_TRIAL}  |  p: {p}  |  T: {T_truncation}")
    print(f"Mean true K+: {true_Ks.mean():.2f}  (range: {true_Ks.min()}–{true_Ks.max()})")
    print(f"Threshold (legacy): {max(OCCUPANCY_THRESHOLD_MIN, OCCUPANCY_THRESHOLD_FRAC*N_PER_TRIAL):.0f}")
    print(f"MC samples for q(K+): {N_MC_SAMPLES}")

    # --- K_+ estimation comparison table ---
    # Each row: (results_key_for_K_hat, display_name)
    estimator_specs = [
        # MFM estimators
        #("mfm_thresh",      "MFM:  threshold"),
        #("mfm_Kmap",        "MFM:  MAP of q(K)"),
        ("mfm_Kplus_mode",  "MFM:  mode of q(K+)"),
        #("mfm_Kplus_round", "MFM:  round(E[K+])"),
        # DP estimators
        #("dp_thresh",       "DP:   threshold"),
        ("dp_Kplus_mode",   "DP:   mode of q(K+)"),
        #("dp_Kplus_round",  "DP:   round(E[K+])"),
        # Finite-T estimators
        #("fin_thresh",      "Fin:  threshold"),
        ("fin_Kplus_mode",  "Fin:  mode of q(K+)"),
        #("fin_Kplus_round", "Fin:  round(E[K+])"),
    ]

    print(f"\n{'Estimator':<25} {'Exact':>7} {'±1':>7} {'MAE':>7} {'RMSE':>7} {'Mean K̂+':>9}")
    print("-" * 70)

    for est_key, est_name in estimator_specs:
        correct = np.array([r[f"correct_{est_key}"] for r in results])
        abs_err = np.array([r[f"abs_error_{est_key}"] for r in results])

        # Recover K_hat values for mean computation
        if est_key.endswith("_thresh"):
            mk = est_key.replace("_thresh", "")
            K_hats = np.array([r[f"k_{mk}_thresh"] for r in results])
        elif est_key.endswith("_Kmap"):
            K_hats = np.array([r["mfm_K_map"] for r in results])
        elif est_key.endswith("_Kplus_mode"):
            mk = est_key.replace("_Kplus_mode", "")
            K_hats = np.array([r[f"{mk}_Kplus_mode"] for r in results])
        elif est_key.endswith("_Kplus_round"):
            mk = est_key.replace("_Kplus_round", "")
            K_hats = np.array([r[f"{mk}_Kplus_round"] for r in results])

        print(f"{est_name:<25} {correct.mean():>7.3f} {(abs_err <= 1).mean():>7.3f} "
              f"{abs_err.mean():>7.3f} {np.sqrt((abs_err**2).mean()):>7.3f} "
              f"{K_hats.mean():>9.2f}")

        
    np.seterr(divide='ignore', invalid='ignore')
    # --- Continuous E[K+] comparison ---
    print(f"\n{'Continuous E_q[K+]':<25} {'Mean':>9} {'Std':>9} {'Corr w/ true':>14}")
    print("-" * 60)
    for mk, name in [("mfm", "MFM"), ("dp", "DP"), ("fin", "Fin")]:
        E_Kp = np.array([r[f"{mk}_E_Kplus"] for r in results])
        corr = np.corrcoef(E_Kp, true_Ks)[0, 1]
        print(f"{name + ':  E_q[K+]':<25} {E_Kp.mean():>9.3f} {E_Kp.std():>9.3f} "
            f"{corr:>14.3f}")
    print(f"{'True K+':<25} {true_Ks.mean():>9.3f} {true_Ks.std():>9.3f}")

    # --- MFM-only: E_q[K] ---
    E_Ks = np.array([r["mfm_E_K"] for r in results])
    corr_EK = np.corrcoef(E_Ks, true_Ks)[0, 1]
    #print(f"\n[MFM only]  Mean E_q[K] = {E_Ks.mean():.3f}  "
    #      f"(std: {E_Ks.std():.3f}, corr w/ true K+: {corr_EK:.3f})")

    # --- Confusion matrices (optional) ---
    if show_confusion:
        for est_key, est_name in estimator_specs:
            if est_key.endswith("_thresh"):
                mk = est_key.replace("_thresh", "")
                K_hats = np.array([r[f"k_{mk}_thresh"] for r in results])
            elif est_key.endswith("_Kmap"):
                K_hats = np.array([r["mfm_K_map"] for r in results])
            elif est_key.endswith("_Kplus_mode"):
                mk = est_key.replace("_Kplus_mode", "")
                K_hats = np.array([r[f"{mk}_Kplus_mode"] for r in results])
            elif est_key.endswith("_Kplus_round"):
                mk = est_key.replace("_Kplus_round", "")
                K_hats = np.array([r[f"{mk}_Kplus_round"] for r in results])

            max_K_seen = max(true_Ks.max(), K_hats.max())
            confusion = np.zeros((max_K_seen, max_K_seen), dtype=int)
            for tk, kh in zip(true_Ks, K_hats):
                confusion[tk - 1, kh - 1] += 1
            print(f"\n[{est_name}] Confusion (rows=true K+, cols=K̂+):")
            header = "        " + "".join(f"{j+1:<5}" for j in range(max_K_seen))
            print(header)
            for i in range(max_K_seen):
                row_vals = "".join(f"{confusion[i, j]:<5}" for j in range(max_K_seen))
                print(f"  {i+1:<5} {row_vals}")

    # --- Parameter recovery ---
    print("\n" + "=" * 85)
    print("PARAMETER RECOVERY")
    print("=" * 85)

    for method_key, method_name in [("mfm", "MFM (proposed)"),
                                     ("dp", "DP-CAVI"),
                                     ("fin", "Finite-T CAVI")]:
        mu_err  = [r[f"{method_key}_mean_mu_error"]
                   for r in results if f"{method_key}_mean_mu_error" in r]
        om_err  = [r[f"{method_key}_mean_omega_error"]
                   for r in results if f"{method_key}_mean_omega_error" in r]
        eta_err = [r[f"{method_key}_mean_eta_error"]
                   for r in results if f"{method_key}_mean_eta_error" in r]
        ari     = [r[f"{method_key}_ari"]
                   for r in results if f"{method_key}_ari" in r]
        nmi     = [r[f"{method_key}_nmi"]
                   for r in results if f"{method_key}_nmi" in r]
        missing = [r[f"{method_key}_n_missing"]
                   for r in results if f"{method_key}_n_missing" in r]
        spurious = [r[f"{method_key}_n_spurious"]
                    for r in results if f"{method_key}_n_spurious" in r]

        print(f"\n[{method_name}]")
        if mu_err:
            print(f"  Mean ||mu_true - mu_est||:         {np.mean(mu_err):.3f}")
            print(f"  Mean ||Omega_true - Omega_est||_F:  {np.mean(om_err):.3f}")
            print(f"  Mean |eta_true - eta_est|:          {np.mean(eta_err):.3f}")
            print(f"  Mean # missing / spurious comps:    "
                  f"{np.mean(missing):.2f} / {np.mean(spurious):.2f}")
            print(f"  Adjusted Rand Index (mean):         {np.mean(ari):.3f}")
            print(f"  Normalized Mutual Info (mean):      {np.mean(nmi):.3f}")
        else:
            print("  (no successful extractions)")

    # --- Per-trial table (optional) ---
    if show_pertrialcomparison:
        print(f"\n{'Tr':<5}{'K+':<5}"
              f"{'MFM':^28}{'DP':^19}{'Fin':^19}")
        print(f"{'':5}{'':5}"
              f"{'thr':<6}{'Kmap':<6}{'mode':<6}{'E[K+]':<10}"
              f"{'thr':<6}{'mode':<6}{'E[K+]':<7}"
              f"{'thr':<6}{'mode':<6}{'E[K+]':<7}")
        print("-" * 90)
        for r in results:
            print(
                f"{r['trial']:<5}{r['true_Kplus']:<5}"
                f"{r['k_mfm_thresh']:<6}{r['mfm_K_map']:<6}"
                f"{r['mfm_Kplus_mode']:<6}{r['mfm_E_Kplus']:<10.2f}"
                f"{r['k_dp_thresh']:<6}{r['dp_Kplus_mode']:<6}"
                f"{r['dp_E_Kplus']:<7.2f}"
                f"{r['k_fin_thresh']:<6}{r['fin_Kplus_mode']:<6}"
                f"{r['fin_E_Kplus']:<7.2f}"
            )