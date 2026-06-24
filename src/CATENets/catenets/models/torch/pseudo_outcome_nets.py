import abc
import copy
from typing import Any, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from src.CATENets.catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CF_FOLDS,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_OUT_T,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_STEP_SIZE_T,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_OUT_T,
    DEFAULT_VAL_SPLIT,
)
from src.CATENets.catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    BasicNetMask,
    BasicNetMask0,
    BasicNetMask1,
    BasicNetMaskHalf,
    PropensityNet,
    PropensityNetMask,
)
from src.CATENets.catenets.models.torch.utils.model_utils import (
    predict_wrapper,
    predict_wrapper_mask,
    train_wrapper,
)
from src.CATENets.catenets.models.torch.utils.transformations import (
    dr_transformation_cate,
    pw_transformation_cate,
    ra_transformation_cate,
    u_transformation_cate,
)


class PseudoOutcomeLearner(BaseCATEEstimator):
    """
    Class implements TwoStepLearners based on pseudo-outcome regression as discussed in
    Curth &vd Schaar (2021): RA-learner, PW-learner and DR-learner

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    po_estimator: sklearn/PyTorch model, default: None
        Custom potential outcome model. If this parameter is set, the rest of the parameters are ignored.
    te_estimator: sklearn/PyTorch model, default: None
        Custom treatment effects model. If this parameter is set, the rest of the parameters are ignored.
    n_folds: int, default 1
        Number of cross-fitting folds. If 1, no cross-fitting
    n_layers_out: int
        First stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out: int
        First stage Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of shared & private representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in representation shared before the hypothesis layers.
    n_layers_out_t: int
        Second stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out_t: int
        Second stage Number of hidden units in each hypothesis layer
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    weight_decay: float
        First stage l2 (ridge) penalty
    weight_decay_t: float
        Second stage l2 (ridge) penalty
    lr: float
        First stage learning rate for optimizer
    lr_: float
        Second stage learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    weighting_strategy: str, default "prop"
        Weighting strategy. Can be "prop" or "1-prop".
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool,
        device: str,
        po_estimator: Any = None,
        te_estimator: Any = None,
        n_folds: int = DEFAULT_CF_FOLDS,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = "prop",
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ):
        super(PseudoOutcomeLearner, self).__init__()
        self.device = device
        self.n_unit_in = n_unit_in
        self.binary_y = binary_y
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.n_layers_out_prop = n_layers_out_prop
        self.weight_decay_t = weight_decay_t
        self.weight_decay = weight_decay
        self.weighting_strategy = weighting_strategy
        self.lr = lr
        self.lr_t = lr_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.nonlin = nonlin
        self.n_folds = n_folds
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.n_layers_out_t = n_layers_out_t
        self.n_units_out_t = n_units_out_t
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        # set estimators
        self._te_template = te_estimator
        self._po_template = po_estimator

        self._te_estimator = self._generate_te_estimator()
        self._po_estimator = self._generate_po_estimator()
        if weighting_strategy is not None:
            self._propensity_estimator = self._generate_propensity_estimator()

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNet(
            name,
            self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=True,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        if self._po_template is not None:
            return copy.deepcopy(self._po_template)

        return BasicNet(
            name,
            self.n_unit_in,
            device=self.device,
            binary_y=self.binary_y,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_propensity_estimator(
        self, name: str = "propensity_estimator"
    ) -> nn.Module:
        if self.weighting_strategy is None:
            raise ValueError("Invalid weighting_strategy for PropensityNet")

        return PropensityNet(
            name,
            self.device,
            self.n_unit_in,
            2,  # number of treatments
            self.weighting_strategy,
            n_units_out_prop=self.n_units_out_prop,
            n_layers_out_prop=self.n_layers_out_prop,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            val_split_prop=self.val_split_prop,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout_prob=self.dropout_prob,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> "PseudoOutcomeLearner":
        """
        Train treatment effects nets.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Train-sample features
        y: array-like of shape (n_samples,)
            Train-sample labels
        w: array-like of shape (n_samples,)
            Train-sample treatments
        """
        self.train()

        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().float()
        w = self._check_tensor(w).squeeze().float()

        n = len(y)

        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(
                X, y, w, pred_mask, pred_mask
            )
        else:
            mu_0_pred, mu_1_pred, p_pred = (
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
            )

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            for train_index, test_index in splitter.split(X.cpu(), w.cpu()):
                # create masks
                pred_mask = torch.zeros(n, dtype=bool).to(self.device)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                (
                    mu_0_pred[pred_mask],
                    mu_1_pred[pred_mask],
                    p_pred[pred_mask],
                ) = self._first_step(X, y, w, ~pred_mask, pred_mask)

        # use estimated propensity scores
        if self.weighting_strategy is not None:
            p = p_pred

        # STEP 2: direct TE estimation
        self._second_step(X, y, w, p, mu_0_pred, mu_1_pred)

        return self

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )
        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        return predict_wrapper(self._te_estimator, X)

    @abc.abstractmethod
    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pass

    def _impute_pos(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # fit two separate (standard) models
        # untreated model
        temp_model_0 = self._generate_po_estimator("po_estimator_0_impute_pos")
        train_wrapper(temp_model_0, X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = self._generate_po_estimator("po_estimator_1_impute_pos")
        train_wrapper(temp_model_1, X_fit[W_fit == 1], Y_fit[W_fit == 1])

        mu_0_pred = predict_wrapper(temp_model_0, X[pred_mask, :])
        mu_1_pred = predict_wrapper(temp_model_1, X[pred_mask, :])

        return mu_0_pred, mu_1_pred

    def _impute_propensity(
        self,
        X: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # split sample
        X_fit, W_fit = X[fit_mask, :], w[fit_mask]

        # fit propensity estimator
        temp_propensity_estimator = self._generate_propensity_estimator(
            "prop_estimator_impute_propensity"
        )
        train_wrapper(temp_propensity_estimator, X_fit, W_fit)

        # predict propensity on hold out
        return temp_propensity_estimator.get_importance_weights(
            X[pred_mask, :], w[pred_mask]
        )

    def _impute_unconditional_mean(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # R-learner and U-learner need to impute unconditional mean
        X_fit, Y_fit = X[fit_mask, :], y[fit_mask]

        # fit model
        temp_model = self._generate_po_estimator("po_est_impute_unconditional_mean")
        train_wrapper(temp_model, X_fit, Y_fit)

        return predict_wrapper(temp_model, X[pred_mask, :])


class PseudoOutcomeLearnerPate(BaseCATEEstimator):
    """
    Class for training with Pate
    """

    def __init__(
        self,
        n_unit_in: int,
        n_unit_in_subset: int,
        binary_y: bool,
        device: str,
        po_estimator: Any = None,
        te_estimator: Any = None,
        n_folds: int = DEFAULT_CF_FOLDS,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = "prop",
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ):
        super(PseudoOutcomeLearnerPate, self).__init__()

        self.n_unit_in = n_unit_in
        self.n_unit_in_subset = n_unit_in_subset
        self.binary_y = binary_y
        self.device = device
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.n_layers_out_prop = n_layers_out_prop
        self.weight_decay_t = weight_decay_t
        self.weight_decay = weight_decay
        self.weighting_strategy = weighting_strategy
        self.lr = lr
        self.lr_t = lr_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.nonlin = nonlin
        self.n_folds = n_folds
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.n_layers_out_t = n_layers_out_t
        self.n_units_out_t = n_units_out_t
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        # set estimators
        self._te_template = te_estimator
        self._po_template = po_estimator

        self._te_estimator = self._generate_te_estimator()
        self._po_estimator = self._generate_po_estimator()
        if weighting_strategy is not None:
            self._propensity_estimator = self._generate_propensity_estimator()

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        if self._po_template is not None:
            return copy.deepcopy(self._po_template)

        return BasicNet(
            name,
            self.n_unit_in,
            device=self.device,
            binary_y=self.binary_y,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_propensity_estimator(
        self, name: str = "propensity_estimator"
    ) -> nn.Module:
        if self.weighting_strategy is None:
            raise ValueError("Invalid weighting_strategy for PropensityNet")

        return PropensityNet(
            name,
            self.device,
            self.n_unit_in,
            2,  # number of treatments
            self.weighting_strategy,
            n_units_out_prop=self.n_units_out_prop,
            n_layers_out_prop=self.n_layers_out_prop,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            val_split_prop=self.val_split_prop,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout_prob=self.dropout_prob,
            dropout=self.dropout,
        ).to(self.device)

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNet(
            name,
            self.n_unit_in_subset,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def fit(
        self, X: torch.Tensor, X_subset: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> "PseudoOutcomeLearnerPate":
        """
        Train treatment effects nets.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Train-sample features
        X_subset: array-like of shape (n_samples, n-1_features)
            Train-sample masekd features
        y: array-like of shape (n_samples,)
            Train-sample labels
        w: array-like of shape (n_samples,)
            Train-sample treatments
        """
        self.train()
        X = self._check_tensor(X).float()
        X_subset = self._check_tensor(X_subset).float()
        y = self._check_tensor(y).squeeze().float()
        w = self._check_tensor(w).squeeze().float()

        n = len(y)
        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(
                X, y, w, pred_mask, pred_mask
            )
        else:
            mu_0_pred, mu_1_pred, p_pred = (
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
            )

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            for train_index, test_index in splitter.split(X.cpu(), w.cpu()):
                # create masks
                pred_mask = torch.zeros(n, dtype=bool).to(self.device)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                (
                    mu_0_pred[pred_mask],
                    mu_1_pred[pred_mask],
                    p_pred[pred_mask],
                ) = self._first_step(X, y, w, ~pred_mask, pred_mask)

        # use estimated propensity scores
        if self.weighting_strategy is not None:
            p = p_pred

        # STEP 2: direct TE estimation, training with subset
        self._second_step(X_subset, y, w, p, mu_0_pred, mu_1_pred)

        return self

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )
        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        return predict_wrapper(self._te_estimator, X)

    @abc.abstractmethod
    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pass

    def _impute_pos(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # fit two separate (standard) models
        # untreated model
        temp_model_0 = self._generate_po_estimator("po_estimator_0_impute_pos")
        train_wrapper(temp_model_0, X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = self._generate_po_estimator("po_estimator_1_impute_pos")
        train_wrapper(temp_model_1, X_fit[W_fit == 1], Y_fit[W_fit == 1])

        mu_0_pred = predict_wrapper(temp_model_0, X[pred_mask, :])
        mu_1_pred = predict_wrapper(temp_model_1, X[pred_mask, :])

        return mu_0_pred, mu_1_pred

    def _impute_propensity(
        self,
        X: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # split sample
        X_fit, W_fit = X[fit_mask, :], w[fit_mask]

        # fit propensity estimator
        temp_propensity_estimator = self._generate_propensity_estimator(
            "prop_estimator_impute_propensity"
        )
        train_wrapper(temp_propensity_estimator, X_fit, W_fit)

        # predict propensity on hold out
        return temp_propensity_estimator.get_importance_weights(
            X[pred_mask, :], w[pred_mask]
        )

    def _impute_unconditional_mean(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # R-learner and U-learner need to impute unconditional mean
        X_fit, Y_fit = X[fit_mask, :], y[fit_mask]

        # fit model
        temp_model = self._generate_po_estimator("po_est_impute_unconditional_mean")
        train_wrapper(temp_model, X_fit, Y_fit)

        return predict_wrapper(temp_model, X[pred_mask, :])


class PseudoOutcomeLearnerMask(BaseCATEEstimator):

    """
    Class implements TwoStepLearners based on pseudo-outcome regression as discussed in
    Curth &vd Schaar (2021): RA-learner, PW-learner and DR-learner

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    po_estimator: sklearn/PyTorch model, default: None
        Custom potential outcome model. If this parameter is set, the rest of the parameters are ignored.
    te_estimator: sklearn/PyTorch model, default: None
        Custom treatment effects model. If this parameter is set, the rest of the parameters are ignored.
    n_folds: int, default 1
        Number of cross-fitting folds. If 1, no cross-fitting
    n_layers_out: int
        First stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out: int
        First stage Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of shared & private representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in representation shared before the hypothesis layers.
    n_layers_out_t: int
        Second stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out_t: int
        Second stage Number of hidden units in each hypothesis layer
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    weight_decay: float
        First stage l2 (ridge) penalty
    weight_decay_t: float
        Second stage l2 (ridge) penalty
    lr: float
        First stage learning rate for optimizer
    lr_: float
        Second stage learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    weighting_strategy: str, default "prop"
        Weighting strategy. Can be "prop" or "1-prop".
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool,
        device: str,
        po_estimator: Any = None,
        te_estimator: Any = None,
        n_folds: int = DEFAULT_CF_FOLDS,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = "prop",
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        mask_dis: str = "Uniform",
    ):
        super(BaseCATEEstimator, self).__init__()
        self.n_unit_in = n_unit_in
        self.binary_y = binary_y
        self.device = device
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.n_layers_out_prop = n_layers_out_prop
        self.weight_decay_t = weight_decay_t
        self.weight_decay = weight_decay
        self.weighting_strategy = weighting_strategy
        self.lr = lr
        self.lr_t = lr_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.nonlin = nonlin
        self.n_folds = n_folds
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.n_layers_out_t = n_layers_out_t
        self.n_units_out_t = n_units_out_t
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.mask_dis = mask_dis
        # set estimators
        self._te_template = te_estimator
        self._po_template = po_estimator

        # if te_estimator is not None:
        #     self._te_estimator = te_estimator
        #     self._add_units()
        # else:

        self._te_estimator = self._generate_te_estimator()
        self._po_estimator = self._generate_po_estimator()

        if weighting_strategy is not None:
            self._propensity_estimator = self._generate_propensity_estimator()

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        if self._po_template is not None:
            return copy.deepcopy(self._po_template)

        return BasicNet(
            name,
            self.n_unit_in,
            device=self.device,
            binary_y=self.binary_y,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_propensity_estimator(
        self, name: str = "propensity_estimator"
    ) -> nn.Module:
        if self.weighting_strategy is None:
            raise ValueError("Invalid weighting_strategy for PropensityNet")

        return PropensityNet(
            name,
            self.device,
            self.n_unit_in,
            2,  # number of treatments
            self.weighting_strategy,
            n_units_out_prop=self.n_units_out_prop,
            n_layers_out_prop=self.n_layers_out_prop,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            val_split_prop=self.val_split_prop,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout_prob=self.dropout_prob,
            dropout=self.dropout,
        ).to(self.device)

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:

        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNetMask(
            name,
            2 * self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
            mask_dis=self.mask_dis,
        ).to(self.device)

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.tensor = None
    ) -> "PseudoOutcomeLearner":
        """
        Train treatment effects nets.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Train-sample features
        y: array-like of shape (n_samples,)
            Train-sample labels
        w: array-like of shape (n_samples,)
            Train-sample treatments
        """
        self.train()

        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().float()
        w = self._check_tensor(w).squeeze().float()

        if b != None:
            b = self._check_tensor(b).squeeze().float()

        n = len(y)

        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(
                X, y, w, pred_mask, pred_mask
            )
        else:
            mu_0_pred, mu_1_pred, p_pred = (
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
            )

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            for train_index, test_index in splitter.split(X.cpu(), w.cpu()):
                # create masks
                pred_mask = torch.zeros(n, dtype=bool).to(self.device)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                (
                    mu_0_pred[pred_mask],
                    mu_1_pred[pred_mask],
                    p_pred[pred_mask],
                ) = self._first_step(X, y, w, ~pred_mask, pred_mask)

        # use estimated propensity scores
        if self.weighting_strategy is not None:
            p = p_pred

        # STEP 2: direct TE estimation
        self._second_step(X, y, w, p, mu_0_pred, mu_1_pred)

        return self

    def predict(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        return_po: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        M: feature masking tensor

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )
        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        M = self._check_tensor(M)
        predict_wrapper_mask(self._te_estimator, X, M)

        return predict_wrapper_mask(self._te_estimator, X, M)

    @abc.abstractmethod
    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pass

    def _impute_pos(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # fit two separate (standard) models
        # untreated model
        temp_model_0 = self._generate_po_estimator("po_estimator_0_impute_pos")
        train_wrapper(temp_model_0, X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = self._generate_po_estimator("po_estimator_1_impute_pos")
        train_wrapper(temp_model_1, X_fit[W_fit == 1], Y_fit[W_fit == 1])

        mu_0_pred = predict_wrapper(temp_model_0, X[pred_mask, :])
        mu_1_pred = predict_wrapper(temp_model_1, X[pred_mask, :])

        return mu_0_pred, mu_1_pred

    def _impute_propensity(
        self,
        X: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # split sample
        X_fit, W_fit = X[fit_mask, :], w[fit_mask]

        # fit propensity estimator
        temp_propensity_estimator = self._generate_propensity_estimator(
            "prop_estimator_impute_propensity"
        )
        train_wrapper(temp_propensity_estimator, X_fit, W_fit)

        # predict propensity on hold out
        return temp_propensity_estimator.get_importance_weights(
            X[pred_mask, :], w[pred_mask]
        )

    def _impute_unconditional_mean(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # R-learner and U-learner need to impute unconditional mean
        X_fit, Y_fit = X[fit_mask, :], y[fit_mask]

        # fit model
        temp_model = self._generate_po_estimator("po_est_impute_unconditional_mean")
        train_wrapper(temp_model, X_fit, Y_fit)

        return predict_wrapper(temp_model, X[pred_mask, :])

    def _add_units(self, pre_trained):
        """
        n_new : integer variable counting the neurons you want to add
        """
        # take a copy of the current weights stored in self._fc which is an
        # ModuleList variable with only one layer
        current = pre_trained.model[0].weight.data.to(self.device)
        output_dim, input_dim = current.shape

        # randomly initialize a tensor with the size of the wanted layer
        hl_input = torch.zeros([output_dim, input_dim]).to(self.device)
        nn.init.xavier_normal_(hl_input, gain=nn.init.calculate_gain(self.nonlin))

        # concatenate the old weights with the new weights
        new_wi = torch.cat([current, hl_input], dim=1)

        # reset weight and grad variables to new size
        self._te_estimator.model[0] = nn.Linear(2 * input_dim, output_dim).to(
            self.device
        )

        # set the weight data to new values
        self._te_estimator.model[0].weight.data = torch.tensor(
            new_wi, requires_grad=True
        )
        # import ipdb;ipdb.set_trace()


class PseudoOutcomeLearnerMaskfull(PseudoOutcomeLearner):
    """
    Class for training with missingness
    """

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNetMask(
            name,
            2 * self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        if self._po_template is not None:
            return copy.deepcopy(self._po_template)

        return BasicNetMask(
            name,
            2 * self.n_unit_in,
            device=self.device,
            binary_y=self.binary_y,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _generate_propensity_estimator(
        self, name: str = "propensity_estimator"
    ) -> nn.Module:
        if self.weighting_strategy is None:
            raise ValueError("Invalid weighting_strategy for PropensityNet")

        return PropensityNetMask(
            name,
            self.device,
            2 * self.n_unit_in,
            2,  # number of treatments
            self.weighting_strategy,
            n_units_out_prop=self.n_units_out_prop,
            n_layers_out_prop=self.n_layers_out_prop,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            val_split_prop=self.val_split_prop,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout_prob=self.dropout_prob,
            dropout=self.dropout,
        ).to(self.device)

    def _impute_pos(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # fit two separate (standard) models
        # untreated model
        temp_model_0 = self._generate_po_estimator("po_estimator_0_impute_pos")
        train_wrapper(temp_model_0, X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = self._generate_po_estimator("po_estimator_1_impute_pos")
        train_wrapper(temp_model_1, X_fit[W_fit == 1], Y_fit[W_fit == 1])

        X = self._check_tensor(X).float()
        M = self._check_tensor(torch.ones(X[pred_mask, :].size()))

        mu_0_pred = predict_wrapper_mask(temp_model_0, X[pred_mask, :], M)
        mu_1_pred = predict_wrapper_mask(temp_model_1, X[pred_mask, :], M)

        return mu_0_pred, mu_1_pred

    def _impute_propensity(
        self,
        X: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        # split sample
        X_fit, W_fit = X[fit_mask, :], w[fit_mask]

        # fit propensity estimator
        temp_propensity_estimator = self._generate_propensity_estimator(
            "prop_estimator_impute_propensity"
        )
        train_wrapper(temp_propensity_estimator, X_fit, W_fit)

        # predict propensity on hold out
        return temp_propensity_estimator.get_importance_weights(
            X[pred_mask, :], w[pred_mask]
        )

    def predict(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        return_po: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        M: feature masking tensor

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )
        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        M = self._check_tensor(M)

        return predict_wrapper_mask(self._te_estimator, X, M)


class DRLearner(PseudoOutcomeLearner):
    """
    DR-learner for CATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerMask(PseudoOutcomeLearnerMask):
    """
    DR-learner for CATE estimation (Training with Masks)
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerMaskFull(PseudoOutcomeLearnerMaskfull):
    """
    DR-learner for PATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerMask1(PseudoOutcomeLearnerMask):
    """
    DR-learner for PATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNetMask1(
            name,
            2 * self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerMask0(PseudoOutcomeLearnerMask):
    """
    DR-learner for PATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNetMask0(
            name,
            2 * self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerPate(PseudoOutcomeLearnerPate):
    """
    DR-learner for PATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred.to(self.device),
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class DRLearnerMaskHalf(DRLearnerMask):
    """
    DR-learner for CATE estimation (Training with Masks)
    """

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        if self._te_template is not None:
            return copy.deepcopy(self._te_template)
        return BasicNetMaskHalf(
            name,
            self.n_unit_in,
            device=self.device,
            binary_y=False,
            n_layers_out=self.n_layers_out_t,
            n_units_out=self.n_units_out_t,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

    def predict(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        return_po: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        M: feature masking tensor

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )
        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        M = self._check_tensor(M).float()

        return predict_wrapper_mask(self._te_estimator, X, M)


class PWLearner(PseudoOutcomeLearner):
    """
    PW-learner for CATE estimation, based on singly robust Horvitz Thompson pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu0_pred, mu1_pred = np.nan, np.nan  # not needed
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return mu0_pred, mu1_pred, p_pred.to(self.device)

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = pw_transformation_cate(y, w, p)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class RALearner(PseudoOutcomeLearner):
    """
    RA-learner for CATE estimation, based on singly robust regression-adjusted pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = ra_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class RALearnerMask(PseudoOutcomeLearnerMask):
    """
    RA-learner for CATE estimation with masking, based on singly robust regression-adjusted pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = ra_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class RALearnerPate(PseudoOutcomeLearnerPate):
    """
    DR-learner for PATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = ra_transformation_cate(y, w, p, mu_0, mu_1)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class ULearner(PseudoOutcomeLearner):
    """
    U-learner for CATE estimation. Based on pseudo-outcome (Y-mu(x))/(w-pi(x))
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu_pred = self._impute_unconditional_mean(X, y, fit_mask, pred_mask).squeeze()
        mu1_pred = np.nan  # only have one thing to impute here
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return mu_pred.to(self.device), mu1_pred, p_pred.to(self.device)

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = u_transformation_cate(y, w, p, mu_0)
        train_wrapper(self._te_estimator, X, pseudo_outcome.detach())


class RLearner(PseudoOutcomeLearner):
    """
    R-learner for CATE estimation. Based on pseudo-outcome (Y-mu(x))/(w-pi(x)) and sample weight
    (w-pi(x))^2 -- can only be implemented if .fit of te_estimator takes argument 'sample_weight'.
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_pred = self._impute_unconditional_mean(X, y, fit_mask, pred_mask).squeeze()
        mu1_pred = np.nan  # only have one thing to impute here
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        return mu_pred.to(self.device), mu1_pred, p_pred.to(self.device)

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = u_transformation_cate(y, w, p, mu_0)
        train_wrapper(
            self._te_estimator, X, pseudo_outcome.detach(), weight=(w - p) ** 2
        )


class XLearner(PseudoOutcomeLearner):
    """
    X-learner for CATE estimation. Combines two CATE estimates via a weighting function g(x):
    tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x)
    """

    def __init__(
        self,
        *args: Any,
        weighting_strategy: str = "prop",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.weighting_strategy = weighting_strategy

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        # split by treatment status, fit one model per group
        pseudo_0 = mu_1[w == 0] - y[w == 0]
        self._te_estimator_0 = self._generate_te_estimator("te_estimator_0_xnet")
        train_wrapper(self._te_estimator_0, X[w == 0], pseudo_0.detach())

        pseudo_1 = y[w == 1] - mu_0[w == 1]
        self._te_estimator_1 = self._generate_te_estimator("te_estimator_1_xnet")
        train_wrapper(self._te_estimator_1, X[w == 1], pseudo_1.detach())

        train_wrapper(self._propensity_estimator, X, w)

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )

        if not training:
            self.eval()

        X = self._check_tensor(X).float().to(self.device)
        tau0_pred = predict_wrapper(self._te_estimator_0, X)
        tau1_pred = predict_wrapper(self._te_estimator_1, X)

        weight = self._propensity_estimator.get_importance_weights(X)
        # import ipdb; ipdb.set_trace()

        return weight[:, None] * tau0_pred + (1 - weight)[:, None] * tau1_pred


class XLearnerMask(PseudoOutcomeLearnerMask):
    """
    X-learner for CATE estimation. Combines two CATE estimates via a weighting function g(x):
    tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x)
    """

    def __init__(
        self,
        *args: Any,
        weighting_strategy: str = "prop",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.weighting_strategy = weighting_strategy

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        # split by treatment status, fit one model per group
        pseudo_0 = mu_1[w == 0] - y[w == 0]
        self._te_estimator_0 = self._generate_te_estimator("te_estimator_0_xnet")
        train_wrapper(self._te_estimator_0, X[w == 0], pseudo_0.detach())

        pseudo_1 = y[w == 1] - mu_0[w == 1]
        self._te_estimator_1 = self._generate_te_estimator("te_estimator_1_xnet")
        train_wrapper(self._te_estimator_1, X[w == 1], pseudo_1.detach())

        train_wrapper(self._propensity_estimator, X, w)

    def predict(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        return_po: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )

        if not training:
            self.eval()

        X = self._check_tensor(X).float()
        M = self._check_tensor(M)

        tau0_pred = predict_wrapper_mask(self._te_estimator_0, X, M)
        tau1_pred = predict_wrapper_mask(self._te_estimator_1, X, M)

        weight = self._propensity_estimator.get_importance_weights(X)

        return weight[:, None] * tau0_pred + (1 - weight)[:, None] * tau1_pred


class XLearnerPate(PseudoOutcomeLearnerPate):
    """
    X-learner for CATE estimation. Combines two CATE estimates via a weighting function g(x):
    tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x)
    """

    def __init__(
        self,
        *args: Any,
        weighting_strategy: str = "prop",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.weighting_strategy = weighting_strategy

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan
        return (
            mu0_pred.squeeze().to(self.device),
            mu1_pred.squeeze().to(self.device),
            p_pred,
        )

    def _second_step(
        self,
        X: torch.Tensor,
        X_subset: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        # split by treatment status, fit one model per group
        pseudo_0 = mu_1[w == 0] - y[w == 0]
        self._te_estimator_0 = self._generate_te_estimator("te_estimator_0_xnet")
        train_wrapper(self._te_estimator_0, X_subset[w == 0], pseudo_0.detach())

        pseudo_1 = y[w == 1] - mu_0[w == 1]
        self._te_estimator_1 = self._generate_te_estimator("te_estimator_1_xnet")
        train_wrapper(self._te_estimator_1, X_subset[w == 1], pseudo_1.detach())

        train_wrapper(self._propensity_estimator, X, w)

    def fit(
        self, X: torch.Tensor, X_subset: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> "XLearnerPate":
        """
        Train treatment effects nets.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Train-sample features
        X_subset: array-like of shape (n_samples, n-1_features)
            Train-sample masekd features
        y: array-like of shape (n_samples,)
            Train-sample labels
        w: array-like of shape (n_samples,)
            Train-sample treatments
        """
        self.train()
        X = self._check_tensor(X).float()
        X_subset = self._check_tensor(X_subset).float()
        y = self._check_tensor(y).squeeze().float()
        w = self._check_tensor(w).squeeze().float()

        n = len(y)
        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(
                X, y, w, pred_mask, pred_mask
            )
        else:
            mu_0_pred, mu_1_pred, p_pred = (
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
                torch.zeros(n).to(self.device),
            )

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            for train_index, test_index in splitter.split(X.cpu(), w.cpu()):
                # create masks
                pred_mask = torch.zeros(n, dtype=bool).to(self.device)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                (
                    mu_0_pred[pred_mask],
                    mu_1_pred[pred_mask],
                    p_pred[pred_mask],
                ) = self._first_step(X, y, w, ~pred_mask, pred_mask)

        # use estimated propensity scores
        if self.weighting_strategy is not None:
            p = p_pred

        # STEP 2: direct TE estimation, training with subset
        self._second_step(X, X_subset, y, w, p, mu_0_pred, mu_1_pred)

        return self

    def predict(
        self,
        X: torch.Tensor,
        X_subset: torch.Tensor,
        return_po: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise NotImplementedError(
                "PseudoOutcomeLearners have no Potential outcome predictors."
            )

        if not training:
            self.eval()

        X = self._check_tensor(X).float().to(self.device)
        X_subset = self._check_tensor(X_subset).float().to(self.device)

        tau0_pred = predict_wrapper(self._te_estimator_0, X_subset)
        tau1_pred = predict_wrapper(self._te_estimator_1, X_subset)

        weight = self._propensity_estimator.get_importance_weights(X)

        return weight[:, None] * tau0_pred + (1 - weight)[:, None] * tau1_pred
