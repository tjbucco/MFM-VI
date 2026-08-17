import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from CAVI.mfm import CAVI_MFM
from CAVI.baselines import DPMixtureCAVI, FiniteMixtureCAVI


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 2024

T_TRUNCATION = 10
MAX_ITER = 500
TOL = 1e-5
VERBOSE = False

STANDARDIZE = True          # Recommended for the Thyroid lab variables
SHOW_CROSSTABS = True

# BNB prior on K, matching your MFM internals
ALPHA_LAMBDA_PRIOR = 1
ALPHA_PI_PRIOR = 4
BETA_PI_PRIOR = 3

# Gamma(shape=A0, rate=B0_GAMMA) prior on alpha_eta
A0 = 1
B0_GAMMA = 1

# NIW prior
LAMBDA0 = 1.0 / 625.0

OCCUPANCY_THRESHOLD_FRAC = 0.02
OCCUPANCY_THRESHOLD_MIN = 4.0


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

from pathlib import Path
import pandas as pd


def load_thyroid_dataframe(local_path="data/thyroid.csv"):
    """
    Load mclust::thyroid without requiring R/rpy2.

    Preferred:
        data/thyroid.csv

    If the local file is absent, try downloading from Rdatasets.
    """

    local_path = Path(local_path)

    if local_path.exists():
        df = pd.read_csv(local_path)
        return df, f"local file: {local_path}"

    # Optional online fallback
    url = (
        "https://raw.githubusercontent.com/vincentarelbundock/"
        "Rdatasets/master/csv/mclust/thyroid.csv"
    )

    try:
        df = pd.read_csv(url)
        return df, "Rdatasets mirror of mclust::thyroid"
    except Exception as e:
        raise RuntimeError(
            "Could not load Thyroid data.\n\n"
            f"Expected local file at: {local_path.resolve()}\n\n"
            "Download it manually from:\n"
            f"{url}\n\n"
            f"Online loading error was:\n{repr(e)}"
        )

def coerce_numeric_columns(df):
    df = df.copy()

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            continue

        nonmissing = df[col].notna()
        converted = pd.to_numeric(df[col].astype(str), errors="coerce")

        if converted[nonmissing].notna().all():
            df[col] = converted

    return df


def prepare_thyroid_data(df, label_col=None):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Rdatasets usually includes a row-name column
    drop_cols = [c for c in ["rownames", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = coerce_numeric_columns(df)

    if label_col is None:
        lower_to_original = {c.lower(): c for c in df.columns}

        for candidate in ["diagnosis", "class", "classification", "group", "type"]:
            if candidate in lower_to_original:
                label_col = lower_to_original[candidate]
                break

    if label_col is None:
        non_numeric = [c for c in df.columns if not is_numeric_dtype(df[c])]
        if len(non_numeric) > 0:
            label_col = non_numeric[0]
        else:
            candidates = [c for c in df.columns if df[c].nunique() <= 10]
            three_level = [c for c in candidates if df[c].nunique() == 3]
            if three_level:
                label_col = three_level[0]
            elif candidates:
                label_col = candidates[0]
            else:
                raise ValueError("Could not infer diagnosis/class column.")

    feature_cols = [
        c for c in df.columns
        if c != label_col and is_numeric_dtype(df[c])
    ]

    if len(feature_cols) != 5:
        print(f"[warning] Expected 5 lab variables, found {len(feature_cols)}:")
        print(feature_cols)

    X = df[feature_cols].to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError("Feature matrix contains missing values.")

    diagnosis = pd.Categorical(df[label_col])
    diagnosis_codes = diagnosis.codes.astype(int)

    if np.any(diagnosis_codes < 0):
        raise ValueError("Diagnosis column contains missing values.")

    diagnosis_names = [str(x) for x in diagnosis.categories]

    return X, diagnosis_codes, diagnosis_names, feature_cols, label_col


# ---------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------

def make_analysis_matrix_and_prior(X, standardize=True):
    """
    Simplified Malsiner-Walli-style NIW analogue.

    If standardizing, data are z-scored and the NIW prior is centered at zero
    with prior mean covariance approximately I, i.e. E[Omega_k] = I.

    If not standardizing, the prior is centered at the empirical mean and uses
    diagonal empirical marginal variances.
    """
    p = X.shape[1]
    nu0 = p + 2.0

    if standardize:
        scaler = StandardScaler()
        Y = scaler.fit_transform(X)
        b0_mu = np.zeros(p)
        target_cov = np.eye(p)
    else:
        scaler = None
        Y = X.astype(float)
        b0_mu = Y.mean(axis=0)
        target_cov = np.diag(Y.var(axis=0, ddof=1))




    B0 = target_cov * (nu0 - p - 1.0)
    
    c0 = 4.5
    g0 = 2.5
    precision_multiplier = 625.0

    b0_mu = np.median(Y, axis=0)
    R = np.ptp(Y, axis=0)

    nu0 = p + 2.0
    B0 = np.diag(R ** 2 / 625.0)


    prior = {
        "lambda0": LAMBDA0,
        "nu0": nu0,
        "b0_mu": b0_mu,
        "B0": B0,
    }

    return Y, prior, scaler


# ---------------------------------------------------------------------
# Responsibility extraction and K inference
# ---------------------------------------------------------------------

def normalize_rows(A):
    A = np.asarray(A, dtype=float)
    A = np.clip(A, 0.0, None)

    row_sums = A.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0

    if np.any(bad):
        A[bad, :] = 1.0 / A.shape[1]
        row_sums = A.sum(axis=1, keepdims=True)

    return A / row_sums


def mfm_marginal_responsibilities(model):
    """
    MFM responsibilities marginalized over q(K).
    """
    N, T = model.N, model.T
    w_nk = np.zeros((N, T))

    kappa_prob = np.asarray(model.kappa_prob, dtype=float).ravel()

    for kappa in range(1, T + 1):
        r = np.asarray(model.r_nk[kappa - 1], dtype=float)

        if r.shape[0] != N and r.shape[1] == N:
            r = r.T

        if r.shape[0] != N:
            raise ValueError(f"Unexpected MFM responsibility shape: {r.shape}")

        w_nk[:, :kappa] += kappa_prob[kappa - 1] * r[:, :kappa]

    return normalize_rows(w_nk)


def get_responsibilities(model):
    """
    Generic extraction for DP and finite-mixture CAVI objects.
    Add your class-specific attribute name here if needed.
    """
    if hasattr(model, "kappa_prob") and hasattr(model, "r_nk"):
        return mfm_marginal_responsibilities(model)

    N = getattr(model, "N", None)

    possible_attrs = [
        "r_nk",
        "responsibilities",
        "resp",
        "phi",
        "tau",
        "varphi",
        "zeta",
        "q_z",
        "qz",
        "gamma_nk",
        "r",
    ]

    for attr in possible_attrs:
        if not hasattr(model, attr):
            continue

        value = getattr(model, attr)

        if isinstance(value, (list, tuple)):
            continue

        try:
            A = np.asarray(value, dtype=float)
        except Exception:
            continue

        if A.ndim != 2:
            continue

        if N is not None:
            if A.shape[0] == N:
                return normalize_rows(A)
            if A.shape[1] == N:
                return normalize_rows(A.T)

    raise AttributeError(
        "Could not find an N x T responsibility matrix. "
        "Add your model's responsibility attribute name to possible_attrs."
    )


def infer_K_from_responsibilities(
    resp,
    threshold_frac=OCCUPANCY_THRESHOLD_FRAC,
    min_count=OCCUPANCY_THRESHOLD_MIN,
):
    N = resp.shape[0]
    effective_counts = resp.sum(axis=0)

    count_threshold = max(min_count, threshold_frac * N)
    active_mask = effective_counts >= count_threshold

    return int(active_mask.sum()), effective_counts, active_mask


def safe_infer_K(model):
    if not hasattr(model, "infer_K"):
        return None, None

    out = model.infer_K()

    if isinstance(out, tuple):
        k_hat = out[0]
        counts = out[1] if len(out) > 1 else None
    else:
        k_hat = out
        counts = None

    try:
        k_hat = int(k_hat)
    except Exception:
        k_hat = None

    return k_hat, counts


# ---------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------

def construct_model(cls, *args, **kwargs):
    """
    Tries to pass b0_mu and B0. If your class version does not accept those
    arguments, falls back to the class defaults.
    """
    try:
        return cls(*args, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "b0_mu" in msg or "B0" in msg:
            kwargs2 = dict(kwargs)
            kwargs2.pop("b0_mu", None)
            kwargs2.pop("B0", None)
            print(f"[warning] {cls.__name__} did not accept b0_mu/B0; using defaults.")
            return cls(*args, **kwargs2)
        raise


def summarize_model(method_name, model, diagnosis_codes, resp=None, k_hat=None, counts=None):
    if resp is None:
        resp = get_responsibilities(model)

    pred = resp.argmax(axis=1)

    if counts is None:
        counts = resp.sum(axis=0)
    else:
        try:
            counts = np.asarray(counts, dtype=float).ravel()
            if counts.size != resp.shape[1]:
                counts = resp.sum(axis=0)
        except Exception:
            counts = resp.sum(axis=0)

    if k_hat is None:
        k_hat, counts, _ = infer_K_from_responsibilities(resp)

    ari = adjusted_rand_score(diagnosis_codes, pred)
    nmi = normalized_mutual_info_score(diagnosis_codes, pred)

    return {
        "method": method_name,
        "k_hat": int(k_hat),
        "counts": counts,
        "pred": pred,
        "ari": ari,
        "nmi": nmi,
    }


def compact_by_size(z):
    order = pd.Series(z).value_counts().index.tolist()
    mapping = {old: i + 1 for i, old in enumerate(order)}
    return np.array([mapping[v] for v in z])


def print_crosstab(summary, diagnosis_codes, diagnosis_names):
    true_names = [diagnosis_names[i] for i in diagnosis_codes]
    pred_compact = compact_by_size(summary["pred"])

    tab = pd.crosstab(
        pd.Series(true_names, name="diagnosis"),
        pd.Series(pred_compact, name="cluster"),
    )

    print(f"\n[{summary['method']}] diagnosis-by-cluster table")
    print(tab.to_string())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)

    df, source = load_thyroid_dataframe()
    X, diagnosis_codes, diagnosis_names, feature_cols, label_col = prepare_thyroid_data(df)

    Y, prior, scaler = make_analysis_matrix_and_prior(X, standardize=STANDARDIZE)

    N, p = Y.shape
    diagnosis_counts = {
        diagnosis_names[i]: int(np.sum(diagnosis_codes == i))
        for i in range(len(diagnosis_names))
    }

    print("\nLoaded Thyroid data")
    print("=" * 70)
    print(f"Source: {source}")
    print(f"N = {N}, p = {p}")
    print(f"Feature columns: {feature_cols}")
    print(f"Diagnosis column: {label_col}")
    print(f"Diagnosis counts: {diagnosis_counts}")
    print(f"Standardized features: {STANDARDIZE}")
    print(f"T truncation: {T_TRUNCATION}")

    # -------------------------------------------------------------
    # 1. Dynamic MFM CAVI
    # -------------------------------------------------------------
    print("\nFitting dynamic MFM CAVI...")

    mfm = construct_model(
        CAVI_MFM,
        Y,
        T=T_TRUNCATION,
        a0=A0,
        b0=B0_GAMMA,
        lambda0=prior["lambda0"],
        nu0=prior["nu0"],
        b0_mu=prior["b0_mu"],
        B0=prior["B0"],
        alpha_lambda_prior=ALPHA_LAMBDA_PRIOR,
        alpha_pi_prior=ALPHA_PI_PRIOR,
        beta_pi_prior=BETA_PI_PRIOR,
    )

    mfm.fit(max_iter=MAX_ITER, tol=TOL, verbose=VERBOSE)

    resp_mfm = mfm_marginal_responsibilities(mfm)
    k_mfm, counts_mfm, _ = infer_K_from_responsibilities(resp_mfm)

    s_mfm = summarize_model(
        "Dynamic MFM-CAVI",
        mfm,
        diagnosis_codes,
        resp=resp_mfm,
        k_hat=k_mfm,
        counts=counts_mfm,
    )

    # -------------------------------------------------------------
    # 2. Truncated DP mixture CAVI
    # -------------------------------------------------------------
    print("Fitting truncated DP mixture CAVI...")

    dp = construct_model(
        DPMixtureCAVI,
        Y,
        T=T_TRUNCATION,
        s1=A0,
        s2=B0_GAMMA,
        lambda0=prior["lambda0"],
        nu0=prior["nu0"],
        b0_mu=prior["b0_mu"],
        B0=prior["B0"],
    )

    dp.fit(max_iter=MAX_ITER, tol=TOL, verbose=VERBOSE)
    k_dp, counts_dp = safe_infer_K(dp)

    s_dp = summarize_model(
        "Truncated DP-CAVI",
        dp,
        diagnosis_codes,
        k_hat=k_dp,
        counts=counts_dp,
    )

    # -------------------------------------------------------------
    # 3. Fixed finite-T mixture CAVI
    # -------------------------------------------------------------
    print("Fitting fixed-T finite mixture CAVI...")

    fin = construct_model(
        FiniteMixtureCAVI,
        Y,
        T=T_TRUNCATION,
        alpha0=A0 / B0_GAMMA,
        lambda0=prior["lambda0"],
        nu0=prior["nu0"],
        b0_mu=prior["b0_mu"],
        B0=prior["B0"],
    )

    fin.fit(max_iter=MAX_ITER, tol=TOL, verbose=VERBOSE)
    k_fin, counts_fin = safe_infer_K(fin)

    s_fin = summarize_model(
        "Fixed-T finite CAVI",
        fin,
        diagnosis_codes,
        k_hat=k_fin,
        counts=counts_fin,
    )

    summaries = [s_mfm, s_dp, s_fin]

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THYROID DATA RESULTS")
    print("=" * 70)
    print("Diagnosis labels are used only for external validation.")
    print(f"Number of diagnosis categories: {len(diagnosis_names)}")

    for s in summaries:
        sorted_counts = np.round(np.sort(s["counts"])[::-1], 1)

        print(f"\n[{s['method']}]")
        print(f"  K_hat: {s['k_hat']}")
        print(f"  ARI vs diagnosis: {s['ari']:.3f}")
        print(f"  NMI vs diagnosis: {s['nmi']:.3f}")
        print(f"  Effective component counts, sorted: {sorted_counts}")

    print("\nMFM posterior q(K):")
    print(np.round(np.asarray(mfm.kappa_prob), 4))

    if SHOW_CROSSTABS:
        for s in summaries:
            print_crosstab(s, diagnosis_codes, diagnosis_names)