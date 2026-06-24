import os
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn import metrics
from src.cate_utils import subgroup_identification

module_path = os.path.abspath(os.path.join("./CATENets/"))

if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models as cate_models
import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
import catenets.models.torch.tlearner as tlearner
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
    NuisanceFunctions,
    attribution_accuracy,
    attribution_insertion_deletion,
    attribution_ranking,
    compute_pehe,
    insertion_deletion,
)


class PropensitySensitivity:
    """
    Sensitivity analysis for confounding.
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
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_val, X_raw_test = load(
            dataset, train_ratio=train_ratio, val_set=True
        )

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type="random",
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        insertion_deletion_results = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
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
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )
            x_val, W_val, Y_val, po0_val, po1_val, _ = sim.simulate_dataset(
                X_raw_val,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )
            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )
            selection_types = ["if_pehe", "pseudo_outcome_r", "pseudo_outcome_dr"]
            log.info("Fitting and explaining learners...")
            learners = {
                "DRLearner": cate_models.torch.DRLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    nonlin="relu",
                ),
                # "CFRNet_0.01": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.01,
                # ),
                # "CFRNet_0.001": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.001,
                # ),
                # "CFRNet_0.0001": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.0001,
                # ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                log.info(f"Fitting {name}.")
                learners[name].fit(X=x_train, y=Y_train, w=W_train)

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=explainer_list,
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    x_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(x_test)

            ## Train nuisance functions
            nuisance_functions = NuisanceFunctions()

            # nuisance_functions.fit(x_train, Y_train, W_train)
            nuisance_functions.fit(x_val, Y_val, W_val)

            for explainer_name in explainer_list:
                for learner_name in learners:
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
                    rank_indices = attribution_ranking(attribution_est)
                    global_rank = np.flip(np.argsort(attribution_est.mean(0)))

                    log.info(f"Calculating Pehe for {explainer_name}. {learner_name}")

                    (
                        insertion_results,
                        deletion_results,
                        insertion_results_truth,
                        deletion_results_truth,
                    ) = insertion_deletion(
                        (
                            x_test[: self.explainer_limit],
                            W_test[: self.explainer_limit],
                            Y_test[: self.explainer_limit],
                        ),
                        x_train.mean(0).reshape(1, -1),
                        rank_indices,
                        learners[learner_name],
                        selection_types,
                        nuisance_functions,
                        cate_test[: self.explainer_limit],
                    )

                    auroc_results = []
                    ate_results = []
                    mse_results = []

                    for feature_idx in range(x_train.shape[1]):

                        auroc, ate, mse = subgroup_identification(
                            global_rank[: feature_idx + 1],
                            x_train,
                            x_test,
                            learners[learner_name],
                        )

                        auroc_results.append(auroc)
                        ate_results.append(ate)
                        mse_results.append(mse)

                    insertion_deletion_results.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            insertion_results,
                            deletion_results,
                            insertion_results_truth,
                            deletion_results_truth,
                            auroc_results,
                            ate_results,
                            mse_results,
                        ]
                    )

                    explainability_data.append(
                        [
                            propensity_scale,
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
                "Propensity Scale",
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
            ],
        )

        results_path = (
            self.save_path
            / f"propensity_sensitivity/insertion_deletion/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )

        with open(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.pkl",
            "wb",
        ) as handle:
            pkl.dump(insertion_deletion_results, handle)


class PropensityAssignment:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type="random",
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        assignment_data = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
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
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            x_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     device= "cuda:1"
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "DRLearner": pseudo_outcome_nets.DRLearner(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=self.batch_size,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     device= "cuda:1"
                # ),
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
                "XLearner": pseudo_outcome_nets.XLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    nonlin="relu",
                ),
                # "CFRNet_0.01": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.01,
                # ),
                # "CFRNet_0.001": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.001,
                # ),
                # "CFRNet_0.0001": cate_models.torch.TARNet(
                #     x_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.0001,
                # ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for name in learners:
                if not "mask" in name.lower():
                    learner_explaintion_lists[name] = [
                        "integrated_gradients",
                        "shapley_value_sampling",
                        "naive_shap",
                    ]
                else:
                    learner_explaintion_lists[name] = ["explain_with_missingness"]
                log.info(f"Fitting {name}.")
                learners[name].fit(X=x_train, y=Y_train, w=W_train)

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(x_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    x_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(x_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:
                    result_auroc = []
                    cate_policy = []
                    cate_random = []
                    cate_original = []

                    true_cate_policy = []
                    true_cate_random = []
                    true_cate_original = []

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
                    if not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=x_test)
                    else:
                        prediction_mask = torch.ones(x_test.shape)
                        cate_pred = learners[learner_name].predict(
                            X=x_test, M=prediction_mask
                        )

                    cate_pred = learners[learner_name].predict(X=x_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    rank_indices = np.argsort(np.mean(attribution_est, axis=0))[::-1]

                    num_feature = num_important_features // 4

                    total_num = len(X_raw_test) + len(X_raw_train)

                    if not "mask" in learner_name.lower():
                        new_y_train = learners[learner_name].predict(X=x_train)
                    else:
                        prediction_mask = torch.ones(x_train.shape)
                        new_y_train = learners[learner_name].predict(
                            X=x_train, M=prediction_mask
                        )

                    # Define threshold & relabel subgroup

                    cate_threshold = torch.mean(new_y_train)
                    new_y_train = (
                        (new_y_train > cate_threshold).float().detach().cpu().numpy()
                    )

                    if not "mask" in learner_name.lower():
                        new_y_test = learners[learner_name].predict(X=x_test)
                    else:
                        prediction_mask = torch.ones(x_test.shape)
                        new_y_test = learners[learner_name].predict(
                            X=x_test, M=prediction_mask
                        )

                    new_y_test = (
                        (new_y_test > cate_threshold).float().detach().cpu().numpy()
                    )

                    # Oracle ATEs

                    true_ites = po1_test - po0_test
                    true_ites_oracle = (
                        np.sum(true_ites[np.where(true_ites > np.mean(true_ites))])
                        / total_num
                    )

                    # ATEs with identified important features
                    xgb_model = xgb.XGBClassifier(
                        objective="binary:logistic", random_state=self.seed
                    )
                    new_x_train = x_train[:, rank_indices[:num_feature]]
                    new_x_test = x_test[:, rank_indices[:num_feature]]

                    xgb_model.fit(new_x_train, new_y_train)

                    y_pred = xgb_model.predict(new_x_test)
                    s_hat = x_test[np.where(y_pred == 1)]

                    result_auroc.append(metrics.roc_auc_score(new_y_test, y_pred))

                    if not "mask" in learner_name.lower():
                        ites = (
                            learners[learner_name]
                            .predict(X=s_hat)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        prediction_mask = torch.ones(s_hat.shape)
                        ites = (
                            learners[learner_name]
                            .predict(X=s_hat, M=prediction_mask)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    cate_policy.append(np.sum(ites) / total_num)
                    true_cate_policy.append(
                        np.sum(true_ites[np.where(y_pred == 1)]) / total_num
                    )

                    # ATEs with random features assignment

                    random_xgb_model = xgb.XGBClassifier(
                        objective="binary:logistic", random_state=self.seed
                    )
                    np.random.shuffle(rank_indices)
                    new_x_train = x_train[:, rank_indices[:num_feature]]
                    new_x_test = x_test[:, rank_indices[:num_feature]]

                    random_xgb_model.fit(new_x_train, new_y_train)
                    y_pred = random_xgb_model.predict(new_x_test)

                    s_hat = x_test[np.where(y_pred == 1)]

                    if len(s_hat) != 0:
                        if not "mask" in learner_name.lower():
                            random_ites = (
                                learners[learner_name]
                                .predict(X=s_hat)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        else:
                            prediction_mask = torch.ones(s_hat.shape)
                            random_ites = (
                                learners[learner_name]
                                .predict(X=s_hat, M=prediction_mask)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        random_ites = 0

                    cate_random.append(np.sum(random_ites) / total_num)
                    true_cate_random.append(
                        np.sum(true_ites[np.where(y_pred == 1)]) / total_num
                    )

                    # ATEs with original assignment

                    s_original = x_test[np.where(W_test == 1), :]

                    if not "mask" in learner_name.lower():
                        original_assignment_ites = (
                            np.sum(
                                learners[learner_name]
                                .predict(X=s_original)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            / total_num
                        )
                    else:
                        prediction_mask = torch.ones(s_original.shape)
                        original_assignment_ites = (
                            np.sum(
                                learners[learner_name]
                                .predict(X=s_original, M=prediction_mask)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            / total_num
                        )

                    cate_original.append(original_assignment_ites)
                    true_cate_original.append(
                        np.sum(true_ites[np.where(W_test == 1)]) / total_num
                    )

                    assignment_data.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            result_auroc,
                            cate_policy,
                            cate_random,
                            cate_original,
                            true_cate_policy,
                            true_cate_random,
                            true_cate_original,
                            true_ites_oracle,
                        ]
                    )

                    explainability_data.append(
                        [
                            propensity_scale,
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
                "Propensity Scale",
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
            ],
        )

        results_path = (
            self.save_path
            / f"propensity_sensitivity/assignment/model_performance/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )
        results_path = (
            self.save_path
            / f"propensity_sensitivity/assignment/assignment/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}"
            f"feature_num_{num_feature}-seed{self.seed}.pkl",
            "wb",
        ) as handle:
            pkl.dump(assignment_data, handle)
