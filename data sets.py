import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def load_old_faithful():
    """Old Faithful eruptions: (eruption duration, waiting time), n=272."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv"
    df = pd.read_csv(url)
    Y = df[["eruptions", "waiting"]].values
    return Y

def load_acidity():
    """Acidity index (log ANC+50), n=155 lakes, univariate."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/mclust/acidity.csv"
    df = pd.read_csv(url)
    # column name may be 'x' or 'acidity' depending on the Rdatasets export --
    # inspect df.columns if this raises a KeyError
    col = [c for c in df.columns if c.lower() not in ("unnamed: 0", "rownames")][0]
    Y = df[[col]].values.astype(float)
    return Y

def load_enzyme():
    """Enzymatic activity, n=245 individuals, univariate."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/mixAK/Enzyme.csv"
    df = pd.read_csv(url)
    col = [c for c in df.columns if c.lower() not in ("unnamed: 0", "rownames")][0]
    Y = df[[col]].values.astype(float)
    return Y

def load_iris_data():
    """Iris: 4 features, n=150, 3 known species (2 overlap)."""
    data = load_iris()
    Y = data.data          # N x 4
    true_labels = data.target  # ground-truth species labels, for ARI etc.
    return Y, true_labels

def generate_highdim_benchmark(n_per_component=5000, p=4, seed=0):
    """
    2-component, well-separated, high-dimensional synthetic benchmark
    (mirrors the setup used in BayesMix's own library benchmarking).
    """
    rng = np.random.default_rng(seed)
    mu1 = np.full(p, 2.0)
    mu2 = -mu1
    cov = np.eye(p)
    Y1 = rng.multivariate_normal(mu1, cov, size=n_per_component)
    Y2 = rng.multivariate_normal(mu2, cov, size=n_per_component)
    Y = np.vstack([Y1, Y2])
    labels = np.concatenate([np.zeros(n_per_component), np.ones(n_per_component)])
    perm = rng.permutation(len(Y))
    return Y[perm], labels[perm].astype(int)


if __name__ == '__main__':
    # Quick check that all datasets load correctly
    for name, loader in [
        ("Old Faithful", load_old_faithful),
        ("Acidity", load_acidity),
        ("Enzyme", load_enzyme),
    ]:
        try:
            Y = loader()
            print(f"{name}: shape={Y.shape}, range=[{Y.min():.2f}, {Y.max():.2f}]")
        except Exception as e:
            print(f"{name}: FAILED to load ({e}) -- check column names / URL")

    Y_iris, labels_iris = load_iris_data()
    print(f"Iris: shape={Y_iris.shape}, true K={len(np.unique(labels_iris))}")

    Y_hd, labels_hd = generate_highdim_benchmark()
    print(f"High-dim synthetic: shape={Y_hd.shape}, true K={len(np.unique(labels_hd))}")