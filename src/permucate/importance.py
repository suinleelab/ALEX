from copy import copy, deepcopy

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.permucate.learners import CateNet, CausalForest
from src.permucate.scoring import (
    compute_pseudo_outcome_risk,
    compute_r_risk,
    compute_tau_risk,
)
from src.permucate.utils import get_learner, get_nuisances_models
from src.cate_utils import init_model
from src.CATENets.catenets.models.torch.pseudo_outcome_nets import PseudoOutcomeLearner

def joblib_compute_conditional_one(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_idx: int,
    fitted_learner,
    learner_type: str,
    importance_estimator,
    n_perm: int,
    random_state: int = 0,
    x_cols: list = None,
    scoring_params: dict = dict(),
    score_fn=None,
    score_ref=None,
    verbose: bool = False,
    groups=None,
    device: str = "cpu",
):
    """Compute a single permucate score for a given column index.

    Parameters
    ----------
    df : pd.DataFrame
        Causal dataset.
    col_idx : int
        Column index for which the residual will be permuted.
    learner : {EconML compatible estimator}
        Cate estimator.
    importance_estimator : {scikit-learn compatible estimator}
        _description_
    n_perm : int
        Number of permutations to compute.
    y_pred_orig : np.ndarray
        Original CATE predictions.
    scoring : str, optional
        Scoring function to use, by default 'r_risk'.
    random_state : int, optional
        Random seed, by default 0.

    """
    rng = np.random.RandomState(random_state)

    if groups is not None:
        group_ids = list(groups[col_idx])
        X_j_train = copy(df_train[x_cols].values[:, group_ids])
        X_minus_j_train = np.delete(df_train[x_cols].values, group_ids, axis=1)
        X_j_test = copy(df_test[x_cols].values[:, group_ids])
        X_minus_j_test = np.delete(df_test[x_cols].values, group_ids, axis=1)
    else:

        X_j_train = copy(df_train[x_cols].values[:, col_idx])
        X_minus_j_train = np.delete(df_train[x_cols].values, col_idx, axis=1)
        X_j_test = copy(df_test[x_cols].values[:, col_idx])
        X_minus_j_test = np.delete(df_test[x_cols].values, col_idx, axis=1)

    # Predict the dependency of X_j on X_minus_j and compute the residuals
    if isinstance(importance_estimator, list):
        importance_estimator_ = importance_estimator[col_idx]
        importance_estimator_.fit(X_minus_j_train, X_j_train)

    elif isinstance(importance_estimator, PseudoOutcomeLearner):
        # Re-fit the entire CATE estimator without the j-th feature
        importance_estimator_ = init_model(
            X_minus_j_train, 
            df_train["y"].values, 
            learner_type, 
            device
        )
        importance_estimator_.fit(
            X_minus_j_train, df_train["y"].values, df_train["a"].values
        )
    else:
        importance_estimator_ = clone(importance_estimator)
        importance_estimator_.fit(X_minus_j_train, X_j_train)
    

    if hasattr(importance_estimator_, "classes_"):
        X_j_hat_test = importance_estimator_.predict_proba(X_minus_j_test)
        X_j_classes = importance_estimator_.classes_
    else:
        X_j_hat_test = importance_estimator_.predict(X_minus_j_test).detach().cpu().numpy().flatten()
        residuals_j_test = X_j_test - X_j_hat_test

    # Predict the CATE with the reconstructed X_j after permuting the residuals
    risk_j_list = []
    for _ in range(n_perm):
        # Classification case, assign the class with the highest probability
        if hasattr(importance_estimator_, "classes_"):
            if isinstance(X_j_hat_test, list):

                X_j_perm = np.stack(
                    [
                        np.array(
                            [
                                rng.choice(X_j_classes[j], size=1, p=X_j_hat_test[j][i])
                                for i in range(X_j_hat_test[j].shape[0])
                            ]
                        )
                        for j in range(len(X_j_hat_test))
                    ]
                )
                X_j_perm = X_j_perm.reshape(X_j_test.shape)
            else:
                X_j_perm = np.array(
                    [
                        rng.choice(X_j_classes, size=1, p=X_j_hat_test[i])
                        for i in range(X_j_hat_test.shape[0])
                    ]
                )
                X_j_perm = X_j_perm.reshape(X_j_test.shape)
        else:
            X_j_perm = X_j_hat_test + rng.permutation(residuals_j_test)

        if groups is not None:
            X_perm = np.empty_like(df_test[x_cols].values)
            X_perm[:, group_ids] = X_j_perm

            X_perm[:, [col for col in range(len(x_cols)) if col not in group_ids]] = (
                X_minus_j_test
            )
            # X_perm = np.insert(X_minus_j_test, group_ids, X_j_perm, axis=1)
        else:
            X_perm = np.insert(X_minus_j_test, col_idx, X_j_perm, axis=1)

        df_test_perm = df_test.copy()
        df_test_perm[x_cols] = X_perm
        risk_j = score_fn(
            cate_estimator=fitted_learner,
            df_test=df_test_perm,
            df_train=df_train,
            x_cols=x_cols,
            **scoring_params,
        )

        risk_j_list.append(risk_j)

    if verbose:
        print(f"risk {col_idx}: {np.array(risk_j).mean()} \n")
        mu_0_mse = mean_squared_error(
            df_test_perm["mu_0"],
            fitted_learner.mu_0.predict(df_test_perm[x_cols].values),
        )
        mu_1_mse = mean_squared_error(
            df_test_perm["mu_1"],
            fitted_learner.mu_1.predict(df_test_perm[x_cols].values),
        )
        print(f"mu_0_mse: {mu_0_mse} \n")
        print(f"mu_1_mse: {mu_1_mse} \n")

    out_dict = {
        "vim": np.array(risk_j_list) - score_ref,
        "nu_j": X_j_hat_test,
    }
    return out_dict


def joblib_compute_permutation_one(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_idx: int,
    fitted_learner,
    n_perm: int,
    random_state: int = 0,
    x_cols: list = None,
    scoring_params: dict = dict(),
    score_fn=None,
    score_ref=None,
    **kwargs,
):
    """Compute a single permucate score for a given column index.

    Parameters
    ----------
    df : pd.DataFrame
        Causal dataset.
    col_idx : int
        Column index for which the residual will be permuted.
    learner : {EconML compatible estimator}
        Cate estimator.
    importance_estimator : {scikit-learn compatible estimator}
        _description_
    n_perm : int
        Number of permutations to compute.
    y_pred_orig : np.ndarray
        Original CATE predictions.
    scoring : str, optional
        Scoring function to use, by default 'r_risk'.
    random_state : int, optional
        Random seed, by default 0.

    """
    rng = np.random.RandomState(random_state)

    X_j_test = copy(df_test[x_cols].values[:, col_idx])
    X_minus_j_test = np.delete(df_test[x_cols].values, col_idx, axis=1)

    # Predict the CATE with the reconstructed X_j after permuting the residuals

    risk_j_list = []
    for _ in range(n_perm):
        X_j_perm = rng.permutation(X_j_test)
        X_perm = np.insert(X_minus_j_test, col_idx, X_j_perm, axis=1)
        df_test_perm = df_test.copy()
        df_test_perm[x_cols] = X_perm
        risk_j_list.append(
            score_fn(
                cate_estimator=fitted_learner,
                df_test=df_test_perm,
                x_cols=x_cols,
                **scoring_params,
            )
        )

    out_dict = {"vim": np.array(risk_j_list) - score_ref}
    return out_dict


def joblib_compute_loco_one(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_idx: int,
    importance_estimator,
    fitted_learner=None,
    learner_type: str = "linear",
    x_cols: list = None,
    scoring_params: dict = dict(),
    score_fn=None,
    score_ref=None,
    verbose: bool = False,
    groups=None,
    device: str = "cpu",
    **kwargs,
):
    out_dict = {}
    if groups is not None:
        group_ids = list(groups[col_idx])
        X_minus_j_train = np.delete(df_train[x_cols].values, group_ids, axis=1)
        df_j_test = df_test.copy()
        cols_gp = [x_cols[c2] for c2 in group_ids]
        df_j_test.drop(columns=cols_gp, inplace=True)
    else:
        X_minus_j_train = np.delete(df_train[x_cols].values, col_idx, axis=1)
        df_j_test = df_test.copy()
        df_j_test.drop(columns=x_cols[col_idx], inplace=True)

    if isinstance(importance_estimator, CateNet) or isinstance(
        importance_estimator, CausalForest
    ):
        # Re-fit the entire CATE estimator without the j-th feature
        importance_estimator_j = clone(importance_estimator)
        importance_estimator_j.fit(
            X=X_minus_j_train, Y=df_train["y"].values, T=df_train["a"].values
        )
    elif isinstance(importance_estimator, PseudoOutcomeLearner):
        # Re-fit the entire CATE estimator without the j-th feature
        importance_estimator_j = init_model(
            X_minus_j_train, 
            df_train["y"].values, 
            learner_type, 
            device
        )
        importance_estimator_j.fit(
            X_minus_j_train, df_train["y"].values, df_train["a"].values
        )
    else:
        # Hines style, only re-fit the final model
        importance_estimator_j = clone(fitted_learner)
        importance_estimator_j.model_final.fit(
            X=X_minus_j_train,
            y=fitted_learner.compute_pseudo_outcomes(
                df_train["y"].values, df_train["a"].values, df_train[x_cols].values
            ),
        )
        if verbose:
            print(
                f"coef {col_idx}: \
            {importance_estimator_j.model_final.coef_} \n"
            )
        if hasattr(importance_estimator_j.model_final, "coef_"):

            out_dict["beta_j"] = importance_estimator_j.model_final.coef_
            out_dict["beta"] = fitted_learner.model_final.coef_

    if groups is not None:
        x_cols_j = [col for col in x_cols if col not in cols_gp]
    else:
        x_cols_j = [col for col in x_cols if col != x_cols[col_idx]]
    risk_j = score_fn(
        cate_estimator=importance_estimator_j,
        df_test=df_j_test,
        df_train=df_train,
        x_cols=x_cols_j,
        **scoring_params,
    )

    out_dict["vim"] = risk_j.ravel() - score_ref.ravel()

    return out_dict


def compute_variable_importance(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    importance_estimator,
    fitted_learner,
    learner_type: str,
    scoring: str = "r_risk",
    x_cols: list = None,
    n_perm: int = 50,
    method: str = "permucate",
    random_state: int = 0,
    n_jobs: int = 1,
    scoring_params: dict = dict(),
    verbose: bool = False,
    return_coefs: bool = False,
    groups=None,
    device: str = "cpu",
):
    """
    Compute variable importance on the test set using the specified method.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset.
    df_test : pd.DataFrame
        Test dataset.
    importance_estimator : {scikit-learn compatible estimator}
        Estimator used to
         - Predict x_j from the other covariates in the permucate method.
         - Predict the CATE from the other covariates in the LOCO method.
    fitted_learner : {EconML compatible estimator}
        Fitted CATE estimator.
    scoring : str, optional
        Scoring function to use, by default 'r_risk'.
    x_cols : list, optional
        List of column names to consider, by default None.
    n_perm : int, optional
        Number of permutations to compute for the permucate method,
        by default 50.
    method : str, optional
        Method to use. Supported methods are ['permucate', 'loco'],
        by default 'permucate'.
    random_state : int, optional
        Random seed, by default 0.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 1.
    """
    if x_cols is None:
        x_cols = df_train.columns[df_train.columns.str.startswith("x")]

    if method == "permucate":
        fit_one = joblib_compute_conditional_one
    elif method == "pi":
        fit_one = joblib_compute_permutation_one
    elif method == "loco":
        fit_one = joblib_compute_loco_one
    else:
        raise ValueError(f"Unknown method: {method}")

    if scoring == "r_risk":
        score_fn = compute_r_risk
    elif scoring == "tau_risk":
        score_fn = compute_tau_risk
    elif scoring == "pseudo_outcome_risk":
        score_fn = compute_pseudo_outcome_risk

    score_ref = score_fn(
        cate_estimator=fitted_learner,
        df_train=df_train,
        df_test=df_test,
        x_cols=x_cols,
        **scoring_params,
    ).reshape(-1)
    if verbose:
        print(f"score_ref: {score_ref.mean()} \n")
    if groups is not None:
        print("Using groups")
        bar = tqdm(range(len(groups)), leave=False, desc="importance j")
    else:
        bar = tqdm(range(len(x_cols)), leave=False, desc="importance j")

    out_list = [
        fit_one(
            df_train=df_train,
            df_test=df_test,
            col_idx=i,
            fitted_learner=fitted_learner,
            importance_estimator=importance_estimator,
            learner_type=learner_type,
            n_perm=n_perm,
            random_state=random_state,
            x_cols=x_cols,
            scoring_params=scoring_params,
            score_fn=score_fn,
            score_ref=score_ref,
            verbose=verbose,
            groups=groups,
            device=device,
        )
        for i in bar
    ]

    vim_list = [out["vim"] for out in out_list]
    if return_coefs and (method == "loco"):
        coefs_j_list = [out["beta_j"] for out in out_list]
        coefs_list = [out["beta"] for out in out_list]
        return np.stack(vim_list), np.stack(coefs_list), np.stack(coefs_j_list)
    elif return_coefs and (method == "permucate"):
        nu_j_list = [out["nu_j"] for out in out_list]
        return np.stack(vim_list), np.stack(nu_j_list)

    return np.stack(vim_list)


def compute_p_val(vim_list, n=None):

    if len(vim_list.shape) == 3:
        vim_list = vim_list.mean(axis=-1)
    mean_cpi = vim_list.mean(axis=-1)
    if n is None:
        z_ = mean_cpi / (
            np.std(vim_list, axis=1) + np.var(vim_list, axis=1) / np.sqrt(n)
        )
    else:
        z_ = mean_cpi / np.std(vim_list, axis=1)
    p_val = norm.sf(z_)

    return p_val


def cross_val_vim(
    df: pd.DataFrame,
    importance_estimator,
    cv,
    model: str,
    meta_learner: str,
    learner_cv=5,
    x_cols: list = None,
    scoring: str = "r_risk",
    n_perm: int = 50,
    method: str = "permucate",
    random_state: int = 0,
    n_jobs: int = 1,
    verbose: bool = False,
    model_nuisances: str = None,
    return_coefs: bool = False,
    **kwargs,
):
    vi_list = []
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)

    for train_idx, test_idx in tqdm(
        cv.split(df, df["a"]), desc="cv", total=cv.get_n_splits()
    ):
        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]

        learner = get_learner(
            model=model,
            meta_learner=meta_learner,
            random_state=0,
            cv=learner_cv,
        )
        learner.fit(
            X=df_train[x_cols].values, Y=df_train["y"].values, T=df_train["a"].values
        )

        # Nuisances
        if meta_learner == "DR":
            pi_hat = learner.model_propensity.predict_proba(df_test[x_cols].values)[
                :, np.argwhere(learner.model_propensity.classes_ == 1.0).ravel()[0]
            ]

            mu_0_hat = learner.mu_0.predict(df_test[x_cols].values)
            mu_1_hat = learner.mu_1.predict(df_test[x_cols].values)
            m_hat = pi_hat * mu_1_hat + (1 - pi_hat) * mu_0_hat
        elif meta_learner == "T":
            _, model_pi = get_nuisances_models(
                n_iter=10, n_jobs=n_jobs, cv=learner_cv, model=model_nuisances
            )
            model_pi.fit(df_train[x_cols].values, df_train["a"].values)
            pi_hat = model_pi.predict_proba(df_test[x_cols].values)[
                :, np.argwhere(model_pi.classes_ == 1.0).ravel()[0]
            ]
            mu_0_hat = learner.mu_0.predict(df_test[x_cols].values)
            mu_1_hat = learner.mu_1.predict(df_test[x_cols].values)
            m_hat = pi_hat * mu_1_hat + (1 - pi_hat) * mu_0_hat

        if method == "loco":
            importance_estimator = deepcopy(learner.model_final)

        vi_list.append(
            compute_variable_importance(
                df_train=df_train,
                df_test=df_test,
                importance_estimator=importance_estimator,
                fitted_learner=learner,
                scoring=scoring,
                x_cols=x_cols,
                n_perm=n_perm,
                method=method,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring_params=dict(
                    m_hat=m_hat,
                    pi_hat=pi_hat,
                    mu_0_hat=mu_0_hat,
                    mu_1_hat=mu_1_hat,
                    tau_true=df_test["tau"].values,
                ),
                return_coefs=return_coefs,
                **kwargs,
            )
        )

    if return_coefs and (method == "loco"):
        vi_concatenated = np.stack([x[0].mean(axis=-1) for x in vi_list], axis=-1)
        coefs_concatenated = np.stack([x[1] for x in vi_list], axis=-1)
        coefs_j_concatenated = np.stack([x[2] for x in vi_list], axis=-1)

        return {
            "vim": vi_concatenated,
            "coefs": coefs_concatenated,
            "coefs_j": coefs_j_concatenated,
        }

    elif return_coefs and (method == "permucate"):
        vi_concatenated = np.stack([x[0].mean(axis=-1) for x in vi_list], axis=-1)
        nu_j_concatenated = np.stack([x[1] for x in vi_list], axis=-1)
        return {"vim": vi_concatenated, "nu_j": nu_j_concatenated}

    vi_concatenated = np.stack([x.mean(axis=-1) for x in vi_list], axis=-1)
    return {"vim": vi_concatenated}