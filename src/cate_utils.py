"""CATE related utility function."""
from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb
from sklearn import metrics
from sklift.metrics import (
    qini_auc_score,
    uplift_at_k,
    uplift_auc_score,
    weighted_average_uplift,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklift.metrics import qini_auc_score

from src.model_utils import NuisanceFunctions, TwoLayerMLP, init_model


def subgroup_identification(
    col_indices: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    cate_model: torch.nn.Module,
    local: bool = False,
) -> tuple:
    """
    Function for calculating evaluationg metrics for treatment assignment in testing set.

    Args:
        rank_indices: global ranking of features
        x_train: training data
        x_test: testing data
        cate_model: trained CATE model.
    Return:
        ate: average treatment effect according to the xgb classifier
        auroc: corresponding AUROC for xgb classifier in testing set.
        mse: mean squared error for student model vs teacher model

    """

    train_effects = cate_model.predict(X=x_train).detach().cpu().numpy()
    test_effects = cate_model.predict(X=x_test).detach().cpu().numpy()

    threshold = np.mean(train_effects)

    train_tx_assignments = train_effects > threshold
    test_tx_assignments = test_effects > threshold

    xgb_model = xgb.XGBClassifier()
    mlp = TwoLayerMLP(len(col_indices), 32, 1)

    if local:
        ## Subgroup identification with local feature ranking.
        assert (
            col_indices.shape == x_test.shape
        ), " local rank indices don't match with testing dimension"

        xgb_model.fit(x_train, train_tx_assignments)
        x_test_copy = np.full(x_test.shape, np.nan)

        for i in range(len(x_test)):
            x_test_copy[i, col_indices[:, i]] = x_test[i, col_indices[:, i]]

        pred_tx_assignment = xgb_model.predict(x_test_copy)

    else:
        ## Subgroup identification with global feature ranking
        assert (
            len(col_indices) <= x_test.shape[1]
        ), " global rank indices don't match with testing dimension"

        mlp.train_model(
            x_train[:, col_indices], train_effects, epochs=100, batch_size=32
        )

        pred_cate = (
            mlp(torch.from_numpy(x_test[:, col_indices]).float()).detach().cpu().numpy()
        )
        mse = np.mean((pred_cate - test_effects) ** 2)

        xgb_model.fit(x_train[:, col_indices], train_tx_assignments)
        pred_tx_assignment = xgb_model.predict(x_test[:, col_indices])

    ate = np.sum(test_effects[pred_tx_assignment == 1]) / len(test_effects)

    auroc = metrics.roc_auc_score(test_tx_assignments, pred_tx_assignment)

    return ate, auroc, mse

def _predict_teacher(est, X, model_type):
    if model_type == "CausalForest":
        y = est.effect(X=X)
    elif model_type == "CATENets":
        y = est.predict(X=X)
    else:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y = est.forward(X=X_t)
    return (y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)).reshape(-1)


def qini_score(
    col_indices: np.ndarray,
    train_data: tuple,
    test_data: tuple,
    teacher,
    model_type: str,
    tau_tr: np.ndarray = None,
    tau_te: np.ndarray = None,
):
    """
    Distill teacher CATE onto top-k features and return:
      train_qini, test_qini, train_mse (student vs teacher), test_mse (student vs teacher)
    """
    x_tr, w_tr, y_tr = train_data
    x_te, w_te, y_te = test_data
    p = x_te.shape[1]

    idx = np.asarray(col_indices, dtype=int).ravel()
    if idx.size == 0 or idx.min() < 0 or idx.max() >= p:
        raise ValueError(f"Invalid feature indices: {idx}")

    # 1) Teacher targets (precompute once outside when calling in a loop)
    if tau_tr is None:
        tau_tr = _predict_teacher(teacher, x_tr, model_type)
    if tau_te is None:
        tau_te = _predict_teacher(teacher, x_te, model_type)

    # 2) Student: Ridge regression on teacher targets (no w_tr / y_tr)
    student = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=1e-2, fit_intercept=True, random_state=0))
    Xtr_k, Xte_k = x_tr[:, idx], x_te[:, idx]
    student.fit(Xtr_k, tau_tr)

    # 3) Student predictions
    pred_tr = student.predict(Xtr_k)
    pred_te = student.predict(Xte_k)

    # 4) Distillation loss (MSE to teacher)
    train_mse = float(np.mean((pred_tr - tau_tr) ** 2))
    test_mse  = float(np.mean((pred_te - tau_te) ** 2))

    # 5) Qini (use student uplift as prediction)
    train_qini = qini_auc_score(y_true=y_tr, uplift=pred_tr, treatment=w_tr)
    test_qini  = qini_auc_score(y_true=y_te, uplift=pred_te, treatment=w_te)

    return train_qini, test_qini, train_mse, test_mse


def qini_score_cal(treatment, outcome, pred_outcome):

    data = list(zip(treatment, outcome, pred_outcome))
    # Sort by descending score
    data_sorted = sorted(data, key=lambda x: x[2], reverse=True)

    # Initialize accumulators
    treated_cumulative_outcome = 0
    control_cumulative_outcome = 0
    treated_count = 0
    qini_values = [0]  # Start with 0 for no one targeted

    for treated, outcome, _ in data_sorted:
        if treated == 1:
            treated_cumulative_outcome += outcome
            treated_count += 1
        else:
            control_cumulative_outcome += outcome

        # Calculate Qini value
        total_treated = sum(treatment)
        if treated_count > 0 and total_treated > 0:
            qini_value = (
                treated_cumulative_outcome / total_treated
                - control_cumulative_outcome / (len(treatment) - total_treated)
            )
            qini_values.append(qini_value)
        else:
            qini_values.append(0)

    # Correcting the end point
    max_qini = (
        treated_cumulative_outcome / total_treated
        - control_cumulative_outcome / (len(treatment) - total_treated)
    )
    qini_values[
        -1
    ] = max_qini  # Ensure the last point reflects the maximum theoretical benefit

    if torch.is_tensor(qini_values):
        qini_values = qini_values.detach().cpu().numpy()
    qini_score = np.trapz(qini_values, np.linspace(0, 1, len(qini_values)))

    return qini_score


def calculate_if_pehe(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    t_plugin: np.ndarray,
    y_test: np.ndarray,
    ident: np.ndarray,
) -> np.ndarray:

    EPS = 1e-7
    a = w_test - p
    c = p * (ident - p)
    b = 2 * np.ones(len(w_test)) * w_test * (w_test - p) / (c + EPS)

    plug_in = (t_plugin - prediction) ** 2
    l_de = (
        (ident - b) * t_plugin ** 2
        + b * y_test * (t_plugin - prediction)
        + (-a * (t_plugin - prediction) ** 2 + prediction ** 2)
    )

    return np.sum(plug_in) + np.sum(l_de)


def calculate_pseudo_outcome_pehe_dr(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    y_test: np.ndarray,
    mu_1: np.ndarray,
    mu_0: np.ndarray,
) -> np.ndarray:

    """
    calculating pseudo outcome for DR
    """

    EPS = 1e-7
    w_1 = w_test / (p + EPS)
    w_0 = (1 - w_test) / (EPS + 1 - p)
    pseudo_outcome = (w_1 - w_0) * y_test + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)

    return np.sqrt(np.mean((prediction - pseudo_outcome) ** 2))


def calculate_pseudo_outcome_pehe_r(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    y_test: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:

    """Calculating pseudo outcome for R"""
    y_pseudo = (y_test - m) - (w_test - p) * prediction

    return np.sqrt(np.mean(y_pseudo ** 2))


def calculate_pehe(
    prediction: np.ndarray,
    test_data: tuple,
    selection_type: str,
    nuisance_functions: NuisanceFunctions,
) -> np.ndarray:

    x_test, w_test, y_test = test_data

    mu_0 = nuisance_functions.predict_mu_0(x_test)
    mu_1 = nuisance_functions.predict_mu_1(x_test)
    mu = nuisance_functions.predict_m(x_test)
    p = nuisance_functions.predict_propensity(x_test)

    t_plugin = mu_1 - mu_0

    ident = np.ones(len(p))
    selection_types = {
        "if_pehe": calculate_if_pehe,
        "pseudo_outcome_dr": calculate_pseudo_outcome_pehe_dr,
        "pseudo_outcome_r": calculate_pseudo_outcome_pehe_r,
    }

    pehe_calculator = selection_types.get(selection_type)

    if pehe_calculator == calculate_if_pehe:
        return pehe_calculator(w_test, p, prediction, t_plugin, y_test, ident)
    elif pehe_calculator == calculate_pseudo_outcome_pehe_dr:
        return pehe_calculator(w_test, p, prediction, y_test, mu_1, mu_0)
    elif pehe_calculator == calculate_pseudo_outcome_pehe_r:
        return pehe_calculator(w_test, p, prediction, y_test, mu)

    raise ValueError(f"Unknown selection_type: {selection_type}")
