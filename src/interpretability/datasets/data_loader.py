import os
import pickle

import numpy as np
from catenets.datasets import load as catenets_load

from src.interpretability.datasets.news.process_news import process_news
from src.interpretability.datasets.tcga.download_and_preprocess import download_and_preprocess

# Project-relative TCGA directory
_TCGA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "tcga")


def normalize_data(X):
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return X_normalized


def load(dataset_name: str, train_ratio: float = 1.0, val_set: bool = False):
    if "tcga" in dataset_name:
        # e.g. dataset_name = "tcga_100" or "tcga_10"
        tcga_path = os.path.join(_TCGA_DIR, dataset_name + ".p")
        # Extract gene count from name (default 100)
        max_genes = int(dataset_name.split("_")[-1]) if "_" in dataset_name else 100
        if not os.path.exists(tcga_path):
            download_and_preprocess(max_genes=max_genes, out_dir=_TCGA_DIR)

        try:
            with open(tcga_path, "rb") as f:
                tcga_dataset = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # Parallel preprocessing can leave a stale/corrupt artifact from older runs.
            # Rebuild once and retry loading.
            if os.path.exists(tcga_path):
                os.remove(tcga_path)
            download_and_preprocess(max_genes=max_genes, out_dir=_TCGA_DIR)
            with open(tcga_path, "rb") as f:
                tcga_dataset = pickle.load(f)
        X_raw = tcga_dataset["rnaseq"]
    elif "news" in dataset_name:
        try:
            news_dataset = pickle.load(
                open(
                    "data/news_100.p",
                    "rb",
                )
            )
        except:
            process_news(max_num_features=100, file_location="data/")
            news_dataset = pickle.load(
                open(
                    "data/" + str(dataset_name) + ".p",
                    "rb",
                )
            )
        X_raw = news_dataset
        # X_raw = 3*(X_raw - np.min(X_raw, axis=0)) / (np.max(X_raw, axis=0) - np.min(X_raw, axis=0)) -1

    elif "twins" in dataset_name:
        # Total features  = 39
        X_raw, _, _, _, _, _ = catenets_load(dataset_name, train_ratio=1.0)
    elif "acic" in dataset_name:
        # Total features  = 55
        X_raw, _, _, _, _, _, _, _ = catenets_load("acic2016")

        X_raw = normalize_data(X_raw)
        # X_raw -= np.mean(X_raw, axis=0)

    else:
        print("Unknown dataset " + str(dataset_name))

    if train_ratio == 1.0:
        return X_raw
    else:
        n = X_raw.shape[0]

        val_idx = int(train_ratio * n)
        train_idx = int(train_ratio * train_ratio * n)

        X_raw_train = X_raw[:train_idx]
        X_raw_val = X_raw[train_idx:val_idx]
        X_raw_test = X_raw[val_idx:]

        train_mean = np.mean(X_raw_train, axis=0)

        X_raw_train -= train_mean
        X_raw_val -= train_mean
        X_raw_test -= train_mean

        if val_set:
            return X_raw_train, X_raw_val, X_raw_test
        else:
            return X_raw_train, X_raw_test
