import os
import pickle as pkl
import sys
from copy import deepcopy
from pathlib import Path

import torch
import xgboost as xgb
from scipy import stats
from sklearn import metrics

# import catenets.models as cate_models

#### import CATE model
module_path = os.path.abspath(os.path.join("./CATENets/"))

if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models as cate_models
import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
import catenets.models.torch.tlearner as tlearner
import numpy as np
import pandas as pd
from catenets.models.jax import (
    DRNet,
    PWNet,
    RANet,
    RNet,
    SNet,
    SNet1,
    SNet2,
    SNet3,
    TNet,
    XNet,
)

import src.interpretability.logger as log
from src.interpretability.datasets.data_loader import load
from src.interpretability.explain import Explainer
from src.interpretability.synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from src.interpretability.utils import (
    attribution_accuracy,
    attribution_insertion_deletion,
    attribution_ranking,
    compute_pehe,
    insertion_deletion,
)


class PredictiveSensitivity:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                x_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                "DRLearner": cate_models.torch.DRLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=self.n_layers,
                    n_units_out=self.n_units_hidden,
                    n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    nonlin="relu",
                ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for learner_name in learners:
                log.info(f"Fitting {learner_name}.")
                learners[learner_name].fit(X=x_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=explainer_list,
                    x_train=x_train,
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[
                    learner_name
                ].explain(x_test[: self.explainer_limit])

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(x_test)

            for learner_name in learners:
                for explainer_name in explainer_list:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    (
                        acc_scores_all_features,
                        acc_scores_all_features_score,
                    ) = attribution_accuracy(all_important_features, attribution_est)
                    (
                        acc_scores_predictive_features,
                        acc_scores_predictive_features_score,
                    ) = attribution_accuracy(pred_features, attribution_est)
                    (
                        acc_scores_prog_features,
                        acc_scores_prog_features_score,
                    ) = attribution_accuracy(prog_features, attribution_est)

                    cate_pred = learners[learner_name].predict(X=x_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score ",
                "Pred features ACC",
                "Pred features ACC Score ",
                "Prog features ACC",
                "Prog features ACC Score ",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / "drlearner/predictive_sensitivity/model_performance"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )


class PredictiveAssignment:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                x_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                # x_train.shape[1],
                # device = "cuda:1",
                # binary_y=(len(np.unique(Y_train)) == 2),
                # n_layers_out=2,
                # n_units_out=100,
                # batch_size=1024,
                # n_iter=self.n_iter,
                # batch_norm=False,
                # nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     x_train.shape[1],
                #     device = "cuda:1",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "TARNet": cate_models.torch.TARNet(
                #      x_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_r=1,
                #      n_layers_out=1,
                #      n_units_out=100,
                #      n_units_r=100,
                #      batch_size=1024,
                #      n_iter=self.n_iter,
                #      batch_norm=False,
                #      nonlin="relu",
                #  ),
                #  "XLearnerMask": cate_models_masks.XLearnerMask(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                "XLearner": cate_models.torch.XLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    lr=1e-3,
                    patience=10,
                    nonlin="relu",
                ),
                #  "DRLearner": cate_models.torch.DRLearner(
                #      x_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),
                #  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      device="cuda:1",
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      batch_size=256,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      mask_dis = "Uniform"
                #      ),
                #  "DRLearnerHalf": pseudo_outcome_nets.DRLearnerMaskHalf(
                #      x_train.shape[1],
                #      device = "cuda:0",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:
                if not "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "integrated_gradients",
                        "shapley_value_sampling",
                        "naive_shap",
                    ]
                elif "half" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = ["shapley_value_sampling"]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")

                # if learner_name == "DRLearnerMask":
                #     pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                #     learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=x_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                    x_train=x_train,
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[
                    learner_name
                ].explain(x_test[: self.explainer_limit])

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(x_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    (
                        acc_scores_all_features,
                        acc_scores_all_features_score,
                    ) = attribution_accuracy(all_important_features, attribution_est)
                    (
                        acc_scores_predictive_features,
                        acc_scores_predictive_features_score,
                    ) = attribution_accuracy(pred_features, attribution_est)
                    (
                        acc_scores_prog_features,
                        acc_scores_prog_features_score,
                    ) = attribution_accuracy(prog_features, attribution_est)
                    # Computing insertion & deletion results

                    cate_pred = learners[learner_name].predict(X=x_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    # Obtain feature global importance rank

                    rank_indices = np.argsort(np.sum(attribution_est, axis=0))[::-1]
                    # Relabel positive outcome

                    new_y_train = learners[learner_name].predict(X=x_train)
                    # setting threshold

                    cate_threshold = torch.mean(new_y_train)
                    new_y_train = (
                        (new_y_train > cate_threshold).float().detach().cpu().numpy()
                    )

                    new_y_test = learners[learner_name].predict(X=x_test)
                    new_y_test = (
                        (new_y_test > cate_threshold).float().detach().cpu().numpy()
                    )

                    xgb_model = xgb.XGBClassifier(
                        objective="binary:logistic", random_state=self.seed
                    )
                    new_x_train = x_train[:, rank_indices[:5]]
                    new_x_test = x_test[:, rank_indices[:5]]
                    xgb_model.fit(new_x_train, new_y_train)

                    y_pred = xgb_model.predict(new_x_test)
                    auroc = metrics.roc_auc_score(new_y_test, y_pred)
                    s_hat = x_test[np.where(y_pred == 1), :]

                    total_num = len(X_raw_test) + len(X_raw_train)
                    utility_index = (
                        np.mean(
                            learners[learner_name]
                            .predict(X=s_hat)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        * len(s_hat)
                        / total_num
                    )

                    # baseline with random features

                    random_xgb_model = xgb.XGBClassifier(
                        objective="binary:logistic", random_state=self.seed
                    )
                    np.random.shuffle(rank_indices)
                    new_x_train = x_train[:, rank_indices[:5]]
                    new_x_test = x_test[:, rank_indices[:5]]

                    random_xgb_model.fit(new_x_train, new_y_train)
                    y_pred = random_xgb_model.predict(new_x_test)

                    random_auroc = metrics.roc_auc_score(new_y_test, y_pred)

                    s_hat_original = x_test[np.where(W_test == 1), :]

                    s_hat = x_test[np.where(y_pred == 1), :]

                    # if len(s_hat) <= 1:
                    #     random_utility_index = 0
                    # else:

                    random_utility_index = (
                        np.mean(
                            learners[learner_name]
                            .predict(X=s_hat)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        * len(s_hat)
                        / total_num
                    )
                    random_assignment_utility_index = (
                        np.mean(
                            learners[learner_name]
                            .predict(X=s_hat_original)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        * len(s_hat)
                        / total_num
                    )

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                            auroc,
                            utility_index,
                            random_utility_index,
                            random_assignment_utility_index,
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
                "AUROC",
                "utility_score",
                "utility_score_random_feature",
                "utility_score_random_assignment",
            ],
        )

        results_path = self.save_path / "predictive_sensitivity/assignment"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )


class PredictiveSensitivityLoss:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1500,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                x_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                # x_train.shape[1],
                # device = "cuda:1",
                # binary_y=(len(np.unique(Y_train)) == 2),
                # n_layers_out=2,
                # n_units_out=100,
                # batch_size=1024,
                # n_iter=self.n_iter,
                # batch_norm=False,
                # nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     x_train.shape[1],
                #     device = "cuda:1",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "TARNet": cate_models.torch.TARNet(
                #      x_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_r=1,
                #      n_layers_out=1,
                #      n_units_out=100,
                #      n_units_r=100,
                #      batch_size=1024,
                #      n_iter=self.n_iter,
                #      batch_norm=False,
                #      nonlin="relu",
                #  ),
                #  "XLearnerMask": cate_models_masks.XLearnerMask(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     device="cuda:1",
                #     n_layers_out=self.n_layers,
                #     n_units_out=self.n_units_hidden,
                #     n_iter=self.n_iter,
                #     batch_size=256,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     mask_dis="Uniform",
                # ),
                # "DRLearnerMaskBeta": pseudo_outcome_nets.DRLearnerMask(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     device="cuda:1",
                #     n_layers_out=self.n_layers,
                #     n_units_out=self.n_units_hidden,
                #     n_iter=self.n_iter,
                #     batch_size=256,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     mask_dis="Beta",
                # ),
                # "DRLearnerMask1": pseudo_outcome_nets.DRLearnerMask1(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     device="cuda:1",
                #     n_layers_out=self.n_layers,
                #     n_units_out=self.n_units_hidden,
                #     n_iter=self.n_iter,
                #     batch_size=256,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     mask_dis="Uniform",
                # ),
                # "DRLearnerMask0": pseudo_outcome_nets.DRLearnerMask0(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     device="cuda:1",
                #     n_layers_out=self.n_layers,
                #     n_units_out=self.n_units_hidden,
                #     n_iter=self.n_iter,
                #     batch_size=256,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     mask_dis="Uniform",
                # ),
                # "DRLearnerHalfMask": pseudo_outcome_nets.DRLearnerMaskHalf(
                #     x_train.shape[1],
                #     device="cuda:0",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=self.n_layers,
                #     n_units_out=self.n_units_hidden,
                #     n_iter=self.n_iter,
                #     lr=1e-3,
                #     patience=10,
                #     batch_size=self.batch_size,
                #     batch_norm=False,
                #     nonlin="relu",
                #     mask_dis="Uniform",
                # ),
                "DRLearner": cate_models.torch.DRLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=self.n_layers,
                    n_units_out=self.n_units_hidden,
                    n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    nonlin="relu",
                ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:

                log.info(f"Fitting {learner_name}.")
                learners[learner_name].fit(X=x_train, y=Y_train, w=W_train)
                cate_test = sim.te(x_test)

                if not "mask" in learner_name.lower():
                    print(learner_name)
                    cate_pred = learners[learner_name].predict(X=x_test)

                else:
                    prediction_mask = torch.ones(x_test.shape)
                    cate_pred = learners[learner_name].predict(
                        X=x_test, M=prediction_mask
                    )

                pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                explainability_data.append(
                    [
                        predictive_scale,
                        learner_name,
                        pehe_test,
                        np.mean(cate_test),
                        np.var(cate_test),
                        pehe_test / np.sqrt(np.var(cate_test)),
                    ]
                )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / "losses/default/predictive_sensitivity/model_performance"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )


class PredictiveSensitivityHeldOutOne:
    """
    Held out one analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2, 5, 10],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        held_out_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                x_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")

            learners = {
                "TLearner": tlearner.TLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=self.batch_size,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                # "TLearnerMask": tlearner.TLearnerMask(
                #     x_train.shape[1],
                #     device="cuda:1",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     x_train.shape[1],
                #     device = "cuda:1",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "TARNet": cate_models.torch.TARNet(
                #      x_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_r=1,
                #      n_layers_out=1,
                #      n_units_out=100,
                #      n_units_r=100,
                #      batch_size=1024,
                #      n_iter=self.n_iter,
                #      batch_norm=False,
                #      nonlin="relu",
                #  ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      lr=1e-3,
                #      patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      x_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                #   "DRLearner": cate_models.torch.DRLearner(
                #       x_train.shape[1],
                #       device = "cuda:1",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(
                #       x_train.shape[1],
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       device="cuda:1",
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       lr=1e-4,
                #       patience=10,
                #       nonlin="relu",
                #       mask_dis="Uniform"
                #       ),
                #   "RALearnerMask": pseudo_outcome_nets.RALearnerMask(
                #       x_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "RALearner": pseudo_outcome_nets.RALearner(
                #       x_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:
                if not "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "integrated_gradients",
                        "shapley_value_sampling",
                        "naive_shap",
                    ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")
                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=x_train, y=Y_train, w=W_train)

                # Obtaining explanation

                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                    x_train=x_train,
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[
                    learner_name
                ].explain(x_test[: self.explainer_limit])

            cate_test = sim.te(x_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    if not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=x_test)
                    else:
                        prediction_mask = torch.ones(x_test.shape)
                        cate_pred = learners[learner_name].predict(
                            X=x_test, M=torch.ones((x_test.shape))
                        )

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            # Held out experiment - iterating x_s = D \ {i} for all i in features.

            for feature_index in range(x_train.shape[1]):

                masks = np.ones((x_train.shape[1]))
                masks[feature_index] = 0

                x_train_subset = x_train[:, masks.astype(bool)]
                x_test_subset = x_test[:, masks.astype(bool)]

                x_train_mask = np.copy(x_train)
                x_train_mask[:, feature_index] = 0

                _, W_train_pate, Y_train_pate, _, _, _, = sim.simulate_dataset(
                    x_train_mask,
                    predictive_scale=predictive_scale,
                    binary_outcome=binary_outcome,
                )

                pate_learners = {
                    "TLearner": tlearner.TLearner(
                        x_train_subset.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=self.n_iter,
                        batch_size=self.batch_size,
                        batch_norm=False,
                        lr=1e-3,
                        nonlin="relu",
                    ),
                    #     "DRLearner": pseudo_outcome_nets.DRLearner(
                    #         x_train_subset.shape[1],
                    #         device="cuda:1",
                    #         binary_y=(len(np.unique(Y_train)) == 2),
                    #         n_layers_out=2,
                    #         n_units_out=100,
                    #         n_iter=self.n_iter,
                    #         batch_size=self.batch_size,
                    #         batch_norm=False,
                    #         lr=1e-3,
                    #         patience=10,
                    #         nonlin="relu"
                    #  )
                }

                subset_explainer_list = [
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    "explain_with_missingness",
                ]

                for learner_name in pate_learners:
                    # model_path = f"models/{dataset}_{num_important_features}/predictive_scale_{predictive_scale}_feature_index_{feature_index}_seed{self.seed}.pt"

                    # try:
                    #     log.info("checking if PATE models"+ learner_name + " exists.")
                    #     pate_learners[learner_name].load_state_dict(torch.load(model_path))
                    #     pate_learners[learner_name].eval()

                    # except:
                    log.info("Fitting PATE models" + learner_name)

                    pate_learners[learner_name].fit(
                        # nuisance function params
                        X=x_train_subset,
                        # X_subset=x_train_subset,
                        y=Y_train,
                        w=W_train,
                    )

                    # torch.save(pate_learners[learner_name].state_dict(), model_path)

                    # pate_learners[learner_name].fit(
                    #     # nuisance function params
                    #     X=x_train_subset,
                    #     y=Y_train,
                    #     w=W_train,
                    # )

                    cate_pred = (
                        learners[learner_name]
                        .predict(X=x_test[: self.explainer_limit])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    pate_pred = (
                        pate_learners[learner_name]
                        .predict(X=x_test_subset[: self.explainer_limit])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    prediction_mask = torch.ones((x_test[: self.explainer_limit].shape))
                    prediction_mask[:, feature_index] = 0
                    # TLearnerMask removed — use pate_pred as fallback
                    pate_mask_pred = pate_pred

                    pate_pehe = np.sqrt(np.square(pate_pred - pate_mask_pred))

                    # loading attribution score from learner_explanations

                    for explainer_name in subset_explainer_list:
                        if explainer_name == "explain_with_missingness":
                            learner_name += "Mask"
                        elif explainer_name == "shapley_value_sampling_half_mask":
                            learner_name = "DRLearnerHalf"
                            explainer_name = "shapley_value_sampling"

                        attribution = learner_explanations[learner_name][
                            explainer_name
                        ][:, feature_index]

                        held_out_data.append(
                            [
                                predictive_scale,
                                feature_index,
                                learner_name,
                                explainer_name,
                                attribution,
                                pate_pred,
                                pate_mask_pred,
                                cate_pred,
                                pate_pehe / np.sqrt(np.var(pate_pred)),
                            ]
                        )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / "held_out/tlearner/predictive_sensitivity/model_performance"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )

        results_path = (
            self.save_path
            / "held_out/tlearner/predictive_sensitivity/held_out/"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl",
            "wb",
        ) as handle:
            pkl.dump(held_out_data, handle)


class PredictiveSensitivityHeldOutOneMask:
    """
    Held out one analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2, 5, 10],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        held_out_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                x_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")

            learners = {
                #        "TLearner": tlearner.TLearner(
                #        x_train.shape[1],
                #        device = "cuda:1",
                #        binary_y=(len(np.unique(Y_train)) == 2),
                #        n_layers_out=2,
                #        n_units_out=100,
                #        batch_size=self.batch_size,
                #        n_iter=self.n_iter,
                #        batch_norm=False,
                #        nonlin="relu",
                #    ),
                #        "TLearnerMask": tlearner.TLearnerMask(
                #        x_train.shape[1],
                #        device = "cuda:1",
                #        binary_y=(len(np.unique(Y_train)) == 2),
                #        n_layers_out=2,
                #        n_units_out=100,
                #        batch_size=self.batch_size,
                #        n_iter=self.n_iter,
                #        batch_norm=False,
                #        nonlin="relu",
                #    ),
                # "SLearner": cate_models.torch.SLearner(
                #     x_train.shape[1],
                #     device = "cuda:1",
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "TARNet": cate_models.torch.TARNet(
                #      x_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_r=1,
                #      n_layers_out=1,
                #      n_units_out=100,
                #      n_units_r=100,
                #      batch_size=1024,
                #      n_iter=self.n_iter,
                #      batch_norm=False,
                #      nonlin="relu",
                #  ),
                # "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #     batch_norm=False,
                #     nonlin="relu",
                #     device="cuda:1",
                # ),
                "XLearner": cate_models.torch.XLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    lr=1e-3,
                    patience=10,
                    nonlin="relu",
                ),
                #   "DRLearner": cate_models.torch.DRLearner(
                #       x_train.shape[1],
                #       device = "cuda:1",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(
                #       x_train.shape[1],
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       device="cuda:1",
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       lr=1e-4,
                #       patience=10,
                #       nonlin="relu",
                #       mask_dis="Uniform"
                #       ),
                #   "RALearnerMask": pseudo_outcome_nets.RALearnerMask(
                #       x_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "RALearner": pseudo_outcome_nets.RALearner(
                #       x_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:
                if not "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "integrated_gradients",
                        "shapley_value_sampling",
                        "naive_shap",
                    ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")
                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=x_train, y=Y_train, w=W_train)

                # Obtaining explanation

                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                    x_train=x_train,
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[
                    learner_name
                ].explain(x_test[: self.explainer_limit])

            cate_test = sim.te(x_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    if not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=x_test)
                    else:
                        prediction_mask = torch.ones(x_test.shape)
                        cate_pred = learners[learner_name].predict(
                            X=x_test, M=torch.ones((x_test.shape))
                        )

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            # Held out experiment - iterating x_s = D \ {i} for all i in features.

            for feature_index in range(x_train.shape[1]):

                x_train_subset = np.copy(x_train)
                x_test_subset = np.copy(x_test)

                x_train_subset[:, feature_index] = 0
                x_test_subset[:, feature_index] = 0

                pate_learners = {
                    "XLearner": pseudo_outcome_nets.XLearner(
                        x_train_subset.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=self.n_iter,
                        batch_size=self.batch_size,
                        batch_norm=False,
                        lr=1e-3,
                        patience=10,
                        nonlin="relu",
                    )
                }

                subset_explainer_list = [
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    "explain_with_missingness",
                ]

                for learner_name in pate_learners:
                    # model_path = f"models/{dataset}_{num_important_features}/predictive_scale_{predictive_scale}_feature_index_{feature_index}_seed{self.seed}.pt"

                    # try:
                    #     log.info("checking if PATE models"+ learner_name + " exists.")
                    #     pate_learners[learner_name].load_state_dict(torch.load(model_path))
                    #     pate_learners[learner_name].eval()

                    # except:
                    log.info("Fitting PATE models" + learner_name)

                    pate_learners[learner_name].fit(
                        X=x_train_subset,
                        y=Y_train,
                        w=W_train,
                    )

                    # torch.save(pate_learners[learner_name].state_dict(), model_path)

                    cate_pred = (
                        learners[learner_name]
                        .predict(X=x_test[: self.explainer_limit])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    pate_pred = (
                        pate_learners[learner_name]
                        .predict(X=x_test_subset[: self.explainer_limit])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    prediction_mask = torch.ones((x_test[: self.explainer_limit].shape))
                    prediction_mask[:, feature_index] = 0
                    # XLearnerMask removed — use pate_pred as fallback
                    pate_mask_pred = pate_pred

                    pate_pehe = np.sqrt(np.square(pate_pred - pate_mask_pred))

                    # loading attribution score from learner_explanations

                    for explainer_name in subset_explainer_list:
                        if explainer_name == "explain_with_missingness":
                            learner_name += "Mask"

                        attribution = learner_explanations[learner_name][
                            explainer_name
                        ][:, feature_index]

                        held_out_data.append(
                            [
                                predictive_scale,
                                feature_index,
                                learner_name,
                                explainer_name,
                                attribution,
                                pate_pred,
                                pate_mask_pred,
                                cate_pred,
                                pate_pehe / np.sqrt(np.var(pate_pred)),
                            ]
                        )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / "held_out_mask/xlearner/predictive_sensitivity/model_performance"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )

        results_path = (
            self.save_path
            / "held_out_mask/xlearner/predictive_sensitivity/held_out/"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl",
            "wb",
        ) as handle:
            pkl.dump(held_out_data, handle)
