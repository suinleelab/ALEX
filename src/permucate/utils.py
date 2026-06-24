import numpy as np
from scipy.stats import norm, t
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.permucate.learners import DRLearner, TLearner


def get_super_learner(
    meta_learner="T",
    n_iter=10,
    random_state=0,
    n_jobs=3,
    cv=3,
    random_search_dict_reg={
        "lr__alpha": np.logspace(-3, 3, 10),
        "dt__learning_rate": np.logspace(-3, 0, 5),
        "dt__max_leaf_nodes": np.arange(10, 100, 5),
    },
    random_search_clf={
        "lr__C": np.logspace(-3, 3, 10),
        "dt__learning_rate": np.logspace(-3, 0, 5),
        "dt__max_leaf_nodes": np.arange(10, 100, 5),
    },
):

    model_reg = RandomizedSearchCV(
        estimator=StackingRegressor(
            estimators=[
                ("lr", Ridge()),
                ("dt", HistGradientBoostingRegressor()),
            ],
            final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv),
        ),
        param_distributions=random_search_dict_reg,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        cv=cv,
    )
    if meta_learner == "T":
        learner = TLearner(models=model_reg)
    elif meta_learner == "DR":
        model_clf = RandomizedSearchCV(
            estimator=StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("dt", HistGradientBoostingClassifier()),
                ],
                final_estimator=LogisticRegression(),
            ),
            param_distributions=random_search_clf,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            cv=cv,
        )
        learner = DRLearner(
            model_final=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv),
            model_propensity=model_clf,
            model_response=model_reg,
            cv=cv,
            random_state=random_state,
        )
    return learner


def get_learner(
    model="linear",
    meta_learner="T",
    random_state=0,
    cv=None,
    final_linear=True,
    **kwargs,
):
    if model == "super_learner":
        return get_super_learner(
            meta_learner=meta_learner, random_state=random_state, cv=cv, **kwargs
        )
    elif model == "linear":
        model_reg = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv)
        model_clf = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10))
    elif model == "rf":
        model_reg = HistGradientBoostingRegressor()
        model_clf = HistGradientBoostingClassifier()
    elif model == "poly":
        model_reg = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=3)),
                ("scaler", StandardScaler()),
                ("lr", RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv)),
            ]
        )
        model_clf = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=3)),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=cv)),
            ]
        )
    if meta_learner == "T":
        learner = TLearner(models=model_reg)
    elif meta_learner == "DR":
        if final_linear:
            model_final = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv)
        else:
            model_final = model_reg
        learner = DRLearner(
            model_final=model_final,
            model_propensity=model_clf,
            model_response=model_reg,
            cv=cv,
            random_state=random_state,
        )
    return learner


def get_nuisances_models(
    model="linear",
    cv=3,
    n_iter=10,
    n_jobs=1,
    random_state=0,
    random_search_dict_reg={
        "lr__alpha": np.logspace(-3, 3, 10),
        "dt__learning_rate": np.logspace(-3, 0, 5),
        "dt__max_leaf_nodes": np.arange(10, 100, 5),
    },
    random_search_clf={
        "lr__C": np.logspace(-3, 3, 10),
        "dt__learning_rate": np.logspace(-3, 0, 5),
        "dt__max_leaf_nodes": np.arange(10, 100, 5),
    },
):
    if model == "super_learner":
        model_m = RandomizedSearchCV(
            estimator=StackingRegressor(
                estimators=[
                    ("lr", Ridge()),
                    ("dt", HistGradientBoostingRegressor()),
                ],
                final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
            ),
            param_distributions=random_search_dict_reg,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            cv=cv,
        )
        model_e = RandomizedSearchCV(
            estimator=StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("dt", HistGradientBoostingClassifier()),
                ],
                final_estimator=LogisticRegression(),
            ),
            param_distributions=random_search_clf,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            cv=cv,
        )
    elif model == "linear":
        model_m = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv)
        model_e = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=cv)
    elif model == "poly":
        model_m = Pipeline(
            [
                ("poly", PolynomialFeatures()),
                ("lr", RidgeCV(alphas=np.logspace(-3, 3, 10), cv=cv)),
            ]
        )
        model_e = Pipeline(
            [
                ("poly", PolynomialFeatures()),
                ("lr", LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=cv)),
            ]
        )
    return model_m, model_e


def corrected_std(differences, test_frac, axis=0):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = differences.shape[axis]
    corrected_var = np.var(differences, ddof=1, axis=axis) * (1 / kr + test_frac)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(
    differences,
    test_frac=0.1 / 0.9,
    axis=0,
):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    df = differences.shape[axis] - 1

    mean = np.mean(differences, axis=axis)
    std = corrected_std(differences, test_frac, axis=axis)
    t_stat = mean / std
    p_val = t.sf(t_stat, df)  # right-tailed t-test
    return t_stat, p_val