from copy import deepcopy

import numpy as np
import pandas as pd
from catenets.models.jax import PseudoOutcomeNet
from econml.dml import CausalForestDML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class CATELearner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass


class CateNet(PseudoOutcomeNet):
    def fit(self, X, Y, T):
        return super().fit(y=Y, w=T, X=X)

    def effect(self, X, **kwargs):
        return self.predict(X, **kwargs).reshape(-1)


class TLearner(CATELearner):
    """
    T-Learner for CATE estimation.

    Parameters
    ----------
    models : scikit-learn compatible estimator
        Model to be used to estimate the response functions for each treatment group.
    """

    def __init__(self, models):
        self.mu_0 = deepcopy(models)
        self.mu_1 = deepcopy(models)
        self.models = models

    def fit(self, X, Y, T):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_0 = X[T == 0]
        Y_0 = Y[T == 0]
        X_1 = X[T == 1]
        Y_1 = Y[T == 1]

        for model, X_, Y_ in tqdm(
            zip([self.mu_0, self.mu_1], [X_0, X_1], [Y_0, Y_1]),
            desc="response functions fit",
        ):
            model.fit(X_, Y_)

    def effect(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if hasattr(self.mu_0, "predict_proba"):
            # If the model is a classifier, we need to use the predict_proba method
            mu_0 = self.mu_0.predict_proba(X)[:, 1]
            mu_1 = self.mu_1.predict_proba(X)[:, 1]
        else:
            mu_0 = self.mu_0.predict(X)
            mu_1 = self.mu_1.predict(X)
        return mu_1 - mu_0


class DRLearner(CATELearner):
    def __init__(
        self,
        model_final,
        model_propensity,
        model_response,
        cv=None,
        random_state=None,
        clip=(0.01, 0.99),
    ):
        self.model_final = model_final
        self.model_propensity = model_propensity
        self.mu_0 = deepcopy(model_response)
        self.mu_1 = deepcopy(model_response)
        self.cv = cv
        self.random_state = random_state
        self.model_response = model_response
        self.clip = clip

    def fit(self, X, Y, T, tot_fit_nuisance=True):
        """
        Fit the DR learner.

        Parameters
        ----------
        X : np.ndarray
            Covariates
        Y : np.ndarray
            Response
        T : np.ndarray
            Treatment
        tot_fit_nuisance : bool
            Whether to fit the nuisance models on the whole dataset or not.

        """

        if isinstance(X, pd.DataFrame):
            X = X.values
            Y = Y.values
            T = T.values
        if self.cv is not None:
            cv = StratifiedKFold(
                n_splits=self.cv, random_state=self.random_state, shuffle=True
            )
            phi_hat = np.empty_like(Y)
            X_final = np.empty_like(X)
            for train_idx, test_idx in tqdm(cv.split(X, T), desc="DR CV"):
                # Compute the pseudo-outcomes in a nested CV scheme. Each
                # pseudo-outcome is computed on the test, which has not been
                # used to fit the nuisance models
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
                self.mu_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
                self.mu_1.fit(X_train[T_train == 1], Y_train[T_train == 1])
                # TODO: keep all fitted models to predict their average for the
                # risk computation
                self.model_propensity.fit(X[train_idx], T[train_idx])
                phi_hat_tmp = self.compute_pseudo_outcomes(Y=Y_test, T=T_test, X=X_test)
                phi_hat[test_idx] = phi_hat_tmp
                X_final[test_idx] = X_test

            # Fit the nuisance models on the whole dataset
            if tot_fit_nuisance:
                self.mu_0.fit(X[T == 0], Y[T == 0])
                self.mu_1.fit(X[T == 1], Y[T == 1])
                self.model_propensity.fit(X, T)

        elif self.cv is None:
            self.mu_0.fit(X[T == 0], Y[T == 0])
            self.mu_1.fit(X[T == 1], Y[T == 1])
            self.model_propensity.fit(X, T)
            phi_hat = self.compute_pseudo_outcomes(Y, T, X)
            X_final = X

        # Regress pseudo-outcomes on X
        self.model_final.fit(X_final, phi_hat)
        self.is_fitted_ = True
        return None

    def effect(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_final.predict(X)

    def compute_pseudo_outcomes(self, Y, T, X):
        if hasattr(self.mu_0, "predict_proba"):
            # If the model is a classifier, we need to use the predict_proba method
            mu_0 = self.mu_0.predict_proba(X)[:, 1]
            mu_1 = self.mu_1.predict_proba(X)[:, 1]
        else:
            mu_0 = self.mu_0.predict(X)
            mu_1 = self.mu_1.predict(X)
        e_hat = self.model_propensity.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, *self.clip)
        mu_a = T * mu_1 + (1 - T) * mu_0
        return (Y - mu_a) * (T - e_hat) / (e_hat * (1 - e_hat)) + mu_1 - mu_0


class CausalForest(CausalForestDML):
    def __init__(self, cv=5, **kwargs):
        self.cv = cv
        super().__init__(cv=5, **kwargs)

    def fit(self, X, Y, T):
        return super().fit(Y=Y, T=T, X=X)

    def effect(self, X):
        return super().effect(X=X).reshape(-1)

    def get_params(self, deep=True):
        return {"cv": self.cv}