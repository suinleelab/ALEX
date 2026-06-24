# stdlib
import random
from typing import Optional

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error

abbrev_dict = {
    "shapley_value_sampling": "SVS",
    "integrated_gradients": "IG",
    "kernel_shap": "SHAP",
    "gradient_shap": "GSHAP",
    "feature_permutation": "FP",
    "feature_ablation": "FA",
    "deeplift": "DL",
    "lime": "LIME",
}

explainer_symbols = {
    "shapley_value_sampling": "D",
    "integrated_gradients": "8",
    "kernel_shap": "s",
    "feature_permutation": "<",
    "feature_ablation": "x",
    "deeplift": "H",
    "lime": ">",
}

cblind_palete = sns.color_palette("colorblind", as_cmap=True)
learner_colors = {
    "SLearner": cblind_palete[0],
    "TLearner": cblind_palete[1],
    "TARNet": cblind_palete[3],
    "CFRNet_0.01": cblind_palete[4],
    "CFRNet_0.001": cblind_palete[6],
    "CFRNet_0.0001": cblind_palete[7],
    "DRLearner": cblind_palete[8],
    "XLearner": cblind_palete[5],
    "Truth": cblind_palete[9],
}


class NuisanceFunctions:
    def __init__(self):

        self.mu0 = xgb.XGBRegressor()
        self.mu1 = xgb.XGBRegressor()
        self.m = xgb.XGBRegressor()

        # self.rf = xgb.XGBClassifier(
        #     # reg_lambda=2,
        #     # max_depth=3,
        #     # colsample_bytree=0.2,
        #     # min_split_loss=10
        # )
        self.rf = LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])

    def fit(self, x_val, Y_val, W_val):

        x0, x1 = x_val[W_val == 0], x_val[W_val == 1]
        y0, y1 = Y_val[W_val == 0], Y_val[W_val == 1]

        self.mu0.fit(x0, y0)
        self.mu1.fit(x1, y1)
        self.m.fit(x_val, Y_val)
        self.rf.fit(x_val, W_val)

    def predict_mu_0(self, x):
        return self.mu0.predict(x)

    def predict_mu_1(self, x):
        return self.mu1.predict(x)

    def predict_propensity(self, x):
        return self.rf.predict_proba(x)[:, 1]

    def predict_m(self, x):
        return self.m.predict(x)


def enable_reproducible_results(seed: int = 42) -> None:
    """
    Set a fixed seed for all the used libraries

    Args:
        seed: int
            The seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def dataframe_line_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    explainers: list,
    learners: list,
    x_logscale: bool = True,
    aggregate: bool = False,
    aggregate_type: str = "mean",
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.set_style("white")
    for learner_name in learners:
        for explainer_name in explainers:
            sub_df = df.loc[
                (df["Learner"] == learner_name) & (df["Explainer"] == explainer_name)
            ]
            if aggregate:
                sub_df = sub_df.groupby(x_axis).agg(aggregate_type).reset_index()
            x_values = sub_df.loc[:, x_axis].values
            y_values = sub_df.loc[:, y_axis].values
            ax.plot(
                x_values,
                y_values,
                color=learner_colors[learner_name],
                marker=explainer_symbols[explainer_name],
            )

    learner_lines = [
        Line2D([0], [0], color=learner_colors[learner_name], lw=2)
        for learner_name in learners
    ]
    explainer_lines = [
        Line2D([0], [0], color="black", marker=explainer_symbols[explainer_name])
        for explainer_name in explainers
    ]

    legend_learners = plt.legend(
        learner_lines, learners, loc="lower left", bbox_to_anchor=(1.04, 0.7)
    )
    legend_explainers = plt.legend(
        explainer_lines,
        [abbrev_dict[explainer_name] for explainer_name in explainers],
        loc="lower left",
        bbox_to_anchor=(1.04, 0),
    )
    plt.subplots_adjust(right=0.75)
    ax.add_artist(legend_learners)
    ax.add_artist(legend_explainers)
    if x_logscale:
        ax.set_xscale("log")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig


def compute_pehe(
    cate_true: np.ndarray,
    cate_pred: torch.Tensor,
) -> tuple:

    if torch.is_tensor(cate_pred):
        cate_pred = cate_pred.detach().cpu().numpy()

    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred))
    return pehe


def compute_cate_metrics(
    cate_true: np.ndarray,
    y_true: np.ndarray,
    w_true: np.ndarray,
    mu0_pred: torch.Tensor,
    mu1_pred: torch.Tensor,
) -> tuple:
    mu0_pred = mu0_pred.detach().cpu().numpy()
    mu1_pred = mu1_pred.detach().cpu().numpy()

    cate_pred = mu1_pred - mu0_pred

    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred))

    y_pred = w_true.reshape(len(cate_true),) * mu1_pred.reshape(len(cate_true),) + (
        1
        - w_true.reshape(
            len(cate_true),
        )
    ) * mu0_pred.reshape(
        len(cate_true),
    )
    factual_rmse = np.sqrt(
        mean_squared_error(
            y_true.reshape(
                len(cate_true),
            ),
            y_pred,
        )
    )
    return pehe, factual_rmse


def attribution_accuracy(
    target_features: list, feature_attributions: np.ndarray
) -> tuple:
    """
    Computes the fraction of the most important features that are truly important
    Args:
        target_features: list of truly important feature indices
        feature_attributions: feature attribution outputted by a feature importance method

    Returns:
        Fraction of the most important features that are truly important
    """

    n_important = len(target_features)  # Number of features that are important
    largest_attribution_idx = torch.topk(
        torch.from_numpy(feature_attributions), n_important
    )[
        1
    ]  # Features with largest attribution
    accuracy = 0  # Attribution accuracy
    accuracy_proportion_abs = 0  # Attribution score accuracy

    for k in range(len(largest_attribution_idx)):
        accuracy += len(np.intersect1d(largest_attribution_idx[k], target_features))

    for k in target_features:
        accuracy_proportion_abs += np.sum(np.abs(feature_attributions[:, k]))

    overlapped_features = accuracy / (len(feature_attributions) * n_important)
    overlapped_features_score = accuracy_proportion_abs / np.sum(
        np.abs(feature_attributions)
    )

    return overlapped_features, overlapped_features_score


def attribution_insertion_deletion(
    x_test: np.ndarray,
    rank_indices: list,
    pate_model,  # expects a model with .predict(X=..., M=...) interface
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with feature subsets by insertion and deletion

    Args:
        x_test: testing set for explanation with insertion and deletion
        feature_attributions: feature attribution outputted by a feature importance method
        pate_model: masking models for PATE estimation.
    Returns:
        results of insertion and deletion of PATE.
    """

    n_samples, n_features = x_test.shape
    deletion_results = np.zeros((n_samples, n_features + 1))
    insertion_results = np.zeros((n_samples, n_features + 1))
    row_indices = [i for i in range(n_samples)]

    removal_mask = torch.ones((n_samples, n_features))

    for rank_index, col_indices in enumerate(rank_indices):

        removal_mask[row_indices, col_indices] = 0.0

        cate_pred_subset = pate_model.predict(X=x_test, M=removal_mask)
        cate_pred_subset = cate_pred_subset.detach().cpu().numpy()
        cate_pred = pate_model.predict(X=x_test, M=torch.ones(x_test.shape))
        cate_pred = cate_pred.detach().cpu().numpy()

        deletion_results[:, 0] = cate_pred.flatten()
        deletion_results[:, rank_index + 1] = cate_pred_subset.flatten()

    # Inserting feature & make prediction with masked model

    insertion_mask = torch.zeros((x_test.shape))

    for rank_index, col_indices in enumerate(rank_indices):

        insertion_mask[row_indices, col_indices] = 1.0

        cate_pred_subset = pate_model.predict(X=x_test, M=insertion_mask)
        cate_pred_subset = cate_pred_subset.detach().cpu().numpy()
        cate_pred = pate_model.predict(X=x_test, M=torch.zeros(x_test.shape))
        cate_pred = cate_pred.detach().cpu().numpy()

        insertion_results[:, 0] = cate_pred.flatten()
        insertion_results[:, rank_index + 1] = cate_pred_subset.flatten()

    return insertion_results, deletion_results


def attribution_ranking(feature_attributions: np.ndarray) -> list:
    """ "
    Compute the ranking of features according to atribution score

    Args:
        feature_attributions: an n x d array of feature attribution scores
    Return:
        a d x n list of indices starting from the highest attribution score
    """

    rank_indices = np.argsort(feature_attributions, axis=1)[:, ::-1]
    rank_indices = list(map(list, zip(*rank_indices)))

    return rank_indices


def insertion_deletion(
    test_data: tuple,
    baseline: np.ndarray,
    rank_indices: list,
    cate_model: torch.nn.Module,
    selection_types: Optional[str],
    nuisance_functions: NuisanceFunctions,
    cate_test: np.ndarray,
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with feature subsets by insertion and deletion

    Args:
        x_test: testing set for explanation with insertion and deletion
        feature_attributions: feature attribution outputted by a feature importance method
        pate_model: masking models for PATE estimation.
    Returns:
        results of insertion and deletion of PATE.
    """
    ## training plugin estimator on

    x_test, _, _ = test_data

    n, d = x_test.shape
    x_test_del = x_test.copy()
    x_test_ins = np.tile(baseline, (n, 1))
    baseline = np.tile(baseline, (n, 1))

    deletion_results = {
        selection_type: np.zeros(d + 1) for selection_type in selection_types
    }
    insertion_results = {
        selection_type: np.zeros(d + 1) for selection_type in selection_types
    }

    deletion_results_truth = np.zeros(d + 1)
    insertion_results_truth = np.zeros(d + 1)

    for rank_index in range(len(rank_indices) + 1):
        if rank_index > 0:  # Skip this on the first iteration
            col_indices = rank_indices[rank_index - 1]

            for i in range(n):

                x_test_ins[i, col_indices[i]] = x_test[i, col_indices[i]]
                x_test_del[i, col_indices[i]] = baseline[i, col_indices[i]]

        cate_pred_subset_ins = (
            cate_model.predict(X=x_test_ins).detach().cpu().numpy().flatten()
        )
        cate_pred_subset_del = (
            cate_model.predict(X=x_test_del).detach().cpu().numpy().flatten()
        )

        for selection_type in selection_types:
            # For the insertion process

            insertion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_ins, test_data, selection_type, nuisance_functions
            )

            # For the deletion process
            deletion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_del, test_data, selection_type, nuisance_functions
            )

        insertion_results_truth[rank_index] = compute_pehe(
            cate_true=cate_test, cate_pred=cate_pred_subset_ins
        )
        deletion_results_truth[rank_index] = compute_pehe(
            cate_true=cate_test, cate_pred=cate_pred_subset_del
        )

    return (
        insertion_results,
        deletion_results,
        insertion_results_truth,
        deletion_results_truth,
    )


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

    """
    calculating pseudo outcome for R
    """

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
