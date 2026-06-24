"""Download and preprocess the TCGA gene-expression dataset.

Downloads the raw pickle from Google Drive (if not already present),
then runs the entropy-based gene selection + log-normalisation +
min-max scaling pipeline for a given number of top genes.

Usage (from project root):
    python src/interpretability/datasets/tcga/download_and_preprocess.py
    python src/interpretability/datasets/tcga/download_and_preprocess.py --max_genes 10
    python src/interpretability/datasets/tcga/download_and_preprocess.py --max_genes 100 --out_dir data/tcga
"""

import argparse
import http.cookiejar
import fcntl
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy


# ── Google Drive file ID for tcga_full_dataset.p ─────────────────────────────
GDRIVE_FILE_ID = "1NveePKQscxJ-VZacOm9MHEAVvPKOQJW8"
RAW_FILENAME = "tcga_full_dataset.p"


def download_raw(out_dir: str) -> str:
    """Download tcga_full_dataset.p from Google Drive if missing."""
    import gdown

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, RAW_FILENAME)

    if os.path.exists(raw_path):
        print(f"Raw TCGA file already exists: {raw_path}")
        return raw_path

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading TCGA dataset from Google Drive → {raw_path}")
    try:
        gdown.download(url, raw_path, quiet=False)
    except http.cookiejar.LoadError as e:
        cookie_path = os.path.expanduser("~/.cache/gdown/cookies.txt")
        print(
            "Warning: gdown cookie cache is invalid; retrying without cookies. "
            f"({e})"
        )
        if os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
            except OSError:
                # Continue even if cache deletion fails; no-cookie mode is enough.
                pass
        gdown.download(url, raw_path, quiet=False, use_cookies=False)

    if not os.path.exists(raw_path):
        raise RuntimeError(
            f"Download failed. Try manually:\n"
            f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing\n"
            f"  and place the file at {raw_path}"
        )
    print(f"Download complete: {raw_path}")
    return raw_path


def _filter_genes(gene_data_df: pd.DataFrame, num_genes: int) -> pd.DataFrame:
    """Select top `num_genes` by entropy of expression values."""
    entropies = gene_data_df.apply(lambda col: entropy(col.values))
    top_genes = entropies.nlargest(num_genes).index
    return gene_data_df[top_genes]


def _normalize_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max normalise each feature to [0, 1]."""
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    rng = xmax - xmin
    rng[rng == 0] = 1.0  # avoid division by zero for constant columns
    return (X - xmin) / rng


def preprocess(raw_path: str, max_genes: int, out_dir: str) -> str:
    """Process raw TCGA pickle → filtered + normalised pickle.

    Steps:
        1. Log-transform gene expression: log(x + 1)
        2. Remove constant-expression genes
        3. Select top `max_genes` by entropy
        4. Min-max normalise to [0, 1]

    Returns path to the saved processed pickle.
    """
    print(f"Loading raw TCGA data from {raw_path} ...")
    with open(raw_path, "rb") as f:
        tcga = pickle.load(f)

    rnaseq = np.array(tcga["rnaseq"])
    print(f"  Raw shape: {rnaseq.shape}")

    # 1. Log normalise
    rnaseq = np.log(rnaseq + 1.0)

    # 2. Drop constant genes
    col_range = rnaseq.max(axis=0) - rnaseq.min(axis=0)
    keep = col_range != 0
    rnaseq = rnaseq[:, keep]
    print(f"  After removing constant genes: {rnaseq.shape}")

    # 3. Entropy-based gene selection
    df = pd.DataFrame(rnaseq, columns=range(rnaseq.shape[1]))
    df_filtered = _filter_genes(df, max_genes)
    print(f"  After selecting top-{max_genes} by entropy: {df_filtered.shape}")

    # 4. Min-max normalise
    tcga["rnaseq"] = _normalize_minmax(df_filtered.values)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tcga_{max_genes}.p")
    tmp_out_path = out_path + ".tmp"
    with open(tmp_out_path, "wb") as f:
        pickle.dump(tcga, f)
    # Atomic replace prevents readers from observing a partially written file.
    os.replace(tmp_out_path, out_path)
    print(f"  Saved processed TCGA ({max_genes} genes) → {out_path}")
    return out_path


def download_and_preprocess(
    max_genes: int = 100,
    out_dir: str = "data/tcga",
) -> str:
    """End-to-end: download (if needed) + preprocess.

    Returns path to the processed pickle.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tcga_{max_genes}.p")
    lock_path = out_path + ".lock"

    # Serialize (re)build across parallel jobs so only one process generates tcga_<N>.p.
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path

        raw_path = download_raw(out_dir)
        return preprocess(raw_path, max_genes, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & preprocess TCGA")
    parser.add_argument("--max_genes", type=int, default=100,
                        help="Number of top genes to keep (default: 100)")
    parser.add_argument("--out_dir", type=str, default="data/tcga",
                        help="Output directory (default: data/tcga)")
    args = parser.parse_args()

    download_and_preprocess(max_genes=args.max_genes, out_dir=args.out_dir)
