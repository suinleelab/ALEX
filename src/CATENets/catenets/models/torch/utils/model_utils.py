"""
Model utils shared across different nets
"""
# Author: Alicia Curth, Bogdan Cebere
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import src.CATENets.catenets.logger as log
from src.CATENets.catenets.models.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def make_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
    device: Optional[str] = None,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    X = X.cpu()
    y = y.cpu()
    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return (
            X_t.to(device),
            y_t.to(device),
            X_val.to(device),
            y_val.to(device),
            VALIDATION_STRING,
        )

    w = w.cpu()
    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return (
        X_t.to(device),
        y_t.to(device),
        w_t.to(device),
        X_val.to(device),
        y_val.to(device),
        w_val.to(device),
        VALIDATION_STRING,
    )


def train_wrapper(
    estimator: Any,
    X: torch.Tensor,
    y: torch.Tensor,
    **kwargs: Any,
) -> None:
    if hasattr(estimator, "train"):
        log.debug(f"Train PyTorch network {estimator}")
        estimator.fit(X, y, **kwargs)
    elif hasattr(estimator, "fit"):
        log.debug(f"Train sklearn estimator {estimator}")
        estimator.fit(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    else:
        raise NotImplementedError(f"Invalid estimator for the {estimator}")


def predict_wrapper(estimator: Any, X: torch.Tensor) -> torch.Tensor:

    if hasattr(estimator, "forward"):
        return estimator(X)

    elif hasattr(estimator, "predict_proba"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict_proba(X_np)[:, 0]  # no event probability

        return torch.Tensor(no_event_proba)
    elif hasattr(estimator, "predict"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict(X_np)

        return torch.Tensor(no_event_proba)
    else:
        raise NotImplementedError(f"Invalid estimator for the {estimator}")


def predict_wrapper_mask(
    estimator: Any, X: torch.Tensor, M: torch.tensor
) -> torch.Tensor:

    if hasattr(estimator, "forward"):
        return estimator(X, M)

    elif hasattr(estimator, "predict_proba"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict_proba(X_np)[:, 0]  # no event probability

        return torch.Tensor(no_event_proba)
    elif hasattr(estimator, "predict"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict(X_np)

        return torch.Tensor(no_event_proba)
    else:
        raise NotImplementedError(f"Invalid estimator for the {estimator}")


def generate_masks(X, mask_dis):

    batch_size = X.shape[0]
    num_features = X.shape[1]

    unif = torch.rand(batch_size, num_features)

    if mask_dis == "Uniform":
        ref = torch.rand(batch_size, 1)
    elif mask_dis == "Beta":
        ref = torch.distributions.Beta(2, 2).rsample(sample_shape=(batch_size, 1))

    # remove all 0s

    masks = (unif > ref).float()
    # zeros = (torch.sum(masks, axis=1) == 0).float().nonzero()

    # masks[zeros] = torch.ones(1,num_features)

    return masks


def generate_perturb_label(x, m):
    """Generate corrupted samples.

    Args:
        m: mask matrix
        x: feature matrix

    Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
    """
    # Parameters
    no, dim = x.size()

    # Randomly (and column-wise) shuffle data
    x_bar = torch.zeros([no, dim])
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def restore_parameters(model, best_model):
    """Move parameters from best model to current model."""
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


def weights_init(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..

    init_type = "normal"

    if classname.find("Linear") != -1:
        if init_type == "normal":
            # Gaussian distribution for initialization
            m.weight.data.normal_(mean=0.0, std=1.0 / np.sqrt(m.in_features))

        elif init_type == "uniform":
            # apply a uniform distribution to the weights and a bias=0

            m.weight.data.uniform_(0.0, 1.0)

        if m.bias is not None:
            m.bias.data.zero_()
