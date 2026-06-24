"""
Calculating feature ranking for explanation methods
on clinical datasets, inlcuding ist3, responder, and massive transfusion
"""
import argparse
import collections
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

import src.CATENets.catenets.models as cate_models
from src.cate_utils import _predict_teacher, qini_score, qini_score_cal
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.dataset import Dataset
from src.interpretability.explain import Explainer
from src.model_utils import NuisanceFunctions
from src.permucate.importance import compute_variable_importance
from src.utils import attribution_ranking, insertion_deletion


class EnsembleTeacher:
    """Ensemble model for CATE"""

    def __init__(self, teachers, model_type):
        self.teachers = teachers
        self.model_type = model_type  # reuse your "learner" strings

    def predict(self, X):
        # mean CATE across teachers
        outs = []
        for t in self.teachers:
            if self.model_type in [
                "SLearner",
                "TLearner",
                "TARNet",
                "CFRNet_0.01",
                "CFRNet_0.001",
                "DRLearner",
                "XLearner",
                "RLearner",
                "RALearner",
            ]:
                y = t.predict(X=X)
            else:
                # fallback to torch forward/call if you add custom teachers
                X_t = torch.as_tensor(X, dtype=torch.float32)
                try:
                    y = t(X_t)
                except TypeError:
                    y = t.forward(X_t)
            y = (
                y.detach().cpu().numpy()
                if isinstance(y, torch.Tensor)
                else np.asarray(y)
            )
            outs.append(y.reshape(-1))
        return np.mean(np.vstack(outs), axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-d", "--dataset", help="Dataset", required=True)
    parser.add_argument(
        "-s", "--shuffle", help="shuffle", default=True, action="store_false"
    )
    parser.add_argument(
        "-t", "--num_trials", help="number of runs ", required=True, type=int
    )
    parser.add_argument(
        "-n",
        "--top_n_features",
        help="how many features to extract",
        required=True,
        type=int,
    )
    parser.add_argument("-l", "--learner", help="learner", required=True)
    parser.add_argument(
        "-b",
        "--zero_baseline",
        help="whether to use zero_baseline",
        default=True,
        action="store_false",
    )

    parser.add_argument("-device", "--device", help="device", required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = args["num_trials"]
    top_n_features = args["top_n_features"]
    shuffle = args["shuffle"]
    learner = args["learner"]
    DEVICE = args["device"]
    zero_baseline = args["zero_baseline"]
    print(zero_baseline)
    print("shuffle dataset: ", shuffle)

    explainer_limit = 1000

    selection_types = ["if_pehe", "pseudo_outcome_r", "pseudo_outcome_dr"]

    data = Dataset(cohort_name, 10)

    x_train, _, _ = data.get_data("train")
    x_test, _, _ = data.get_data("test")

    feature_size = x_train.shape[1]

    explainers = [
        # Global methods
        "loco",
        "permucate",
        # Local methods
        "random",
        "saliency",
        "smooth_grad",
        # "gradient_shap",
        "lime",
        "baseline_lime",
        "baseline_shapley_value_sampling",
        "marginal_shapley_value_sampling",
        "integrated_gradients",
        "baseline_integrated_gradients",
        "kernel_shap"
        # "marginal_shap"
    ]

    results_train = np.zeros((trials, len(x_train)))
    results_test = np.zeros((trials, len(x_test)))

    data = Dataset(cohort_name, 0, True)
    teachers = []

    for i in range(trials):

        data = Dataset(cohort_name, i)
        x_train, w_train, y_train = data.get_data("train")
        x_val, w_val, y_val = data.get_data("val")
        x_test, w_test, y_test = data.get_data("test")

        # np.random.seed(i)
        # train_indices = np.random.choice(
        #     len(x_train), size=len(x_train), replace=True
        # )
        # val_indices = np.random.choice(
        #     len(x_val), size=len(x_val), replace=True
        # )
        # test_indices = np.random.choice(
        #     len(x_test), size=len(x_test), replace=True
        # )

        # x_train, y_train, w_train = x_train[train_indices],
        # y_train[train_indices], w_train[train_indices]
        # x_val, y_val, w_val = x_val[val_indices],
        # y_val[val_indices], w_val[val_indices]
        # x_test, y_test, w_test = x_test[test_indices],
        # y_test[test_indices], w_test[test_indices]

        models = {
            # "XLearner": pseudo_outcome_nets.XLearner(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     nonlin="relu",
            #     device=DEVICE,
            #     seed=i,
            # ),
            # "SLearner": cate_models.torch.SLearner(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     nonlin="relu",
            #     device=DEVICE,
            #     seed=i,
            # ),
            # "RLearner": pseudo_outcome_nets.RLearner(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     n_iter=1000,
            #     lr=1e-3,
            #     patience=10,
            #     batch_size=128,
            #     batch_norm=False,
            #     nonlin="relu",
            #     device=DEVICE,
            #     seed=i,
            # ),
            # "RALearner": pseudo_outcome_nets.RALearner(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     n_iter=1000,
            #     lr=1e-3,
            #     patience=10,
            #     batch_size=128,
            #     batch_norm=False,
            #     nonlin="relu",
            #     device=DEVICE,
            #     seed=i,
            # ),
            # "TLearner": cate_models.torch.TLearner(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     nonlin="relu",
            #     device=DEVICE,
            # ),
            # "TARNet": cate_models.torch.TARNet(
            #     x_train.shape[1],
            #     binary_y=True,
            #     n_layers_r=1,
            #     n_layers_out=1,
            #     n_units_out=100,
            #     n_units_r=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     batch_norm=False,
            #     early_stopping=True,
            #     nonlin="relu",
            # ),
            # "CFRNet_0.01": cate_models.torch.TARNet(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_r=2,
            #     n_layers_out=2,
            #     n_units_out=100,
            #     n_units_r=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     lr=1e-3,
            #     batch_norm=False,
            #     nonlin="relu",
            #     penalty_disc=0.01,
            # ),
            # "CFRNet_0.001": cate_models.torch.TARNet(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_r=2,
            #     n_layers_out=2,
            #     n_units_out=100,
            #     n_units_r=100,
            #     lr=1e-5,
            #     batch_size=128,
            #     n_iter=1000,
            #     batch_norm=False,
            #     nonlin="relu",
            #     penalty_disc=0.001,
            #     seed=i,
            # ),
            "DRLearner": pseudo_outcome_nets.DRLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
        }

        learner_explanations = {}
        learner_explainers = {}
        insertion_deletion_data = []

        # Training nuisance function for pehe.

        if data.cohort_name in ["crash_2", "ist3", "sprint", "accord", "accord_glycemia"]:
            nuisance_functions = NuisanceFunctions(rct=True)
        else:
            nuisance_functions = NuisanceFunctions(rct=False)

        nuisance_functions.fit(x_val, y_val, w_val)

        model = models[learner]
        del models  # free memory for the 8 unused models
        baseline = np.median(x_train, axis=0)

        for _, idx_lst in data.discrete_indices.items():
            if len(idx_lst) == 1:

                # setting binary vars to 0.5
                baseline[idx_lst] = 0.5
            else:
                # setting categorical baseline to 1/n
                # category_counts = data[:, idx_lst].sum(axis=0)
                # baseline[idx_lst] = category_counts / category_counts.sum()

                baseline[idx_lst] = 1 / len(idx_lst)

        model.fit(x_train, y_train, w_train)
        teachers.append(model)

        results_train[i] = model.predict(X=x_train).detach().cpu().numpy().flatten()
        results_test[i] = model.predict(X=x_test).detach().cpu().numpy().flatten()

        print(f"Explaining dataset with: {learner}")

        # Explain CATE
        learner_explainers[learner] = Explainer(
            model,
            feature_names=list(range(x_train.shape[1])),
            explainer_list=explainers,
            perturbations_per_eval=1,
            baseline=baseline.reshape(1, -1),
        )
        learner_explanations[learner] = learner_explainers[learner].explain(
            x_test[:explainer_limit], w_test[:explainer_limit], y_test[:explainer_limit]
        )

        # Precompute nuisance quantities shared by loco and permucate
        x_cols = [f"x{j}" for j in range(feature_size)]
        df_train = pd.DataFrame(x_train, columns=x_cols)
        df_train["y"] = y_train
        df_train["a"] = w_train
        df_test_vim = pd.DataFrame(x_test, columns=x_cols)
        df_test_vim["y"] = y_test
        df_test_vim["a"] = w_test
        pi_hat = nuisance_functions.predict_propensity(x_test)
        mu0_hat = nuisance_functions.predict_mu_0(x_test)
        mu1_hat = nuisance_functions.predict_mu_1(x_test)
        m_hat = pi_hat * mu1_hat + (1 - pi_hat) * mu0_hat

        for global_att in explainers:
            if global_att in ["loco", "permucate"]:
                vim = compute_variable_importance(
                    df_train=df_train,
                    df_test=df_test_vim,
                    importance_estimator=model,
                    fitted_learner=model,
                    learner_type=learner,
                    method=global_att,
                    scoring="r_risk",
                    x_cols=x_cols,
                    scoring_params=dict(
                        m_hat=m_hat, pi_hat=pi_hat, mu_0_hat=mu0_hat, mu_1_hat=mu1_hat
                    ),
                    device=DEVICE,
                )
                if global_att == "permucate":
                    learner_explanations[learner][global_att] = vim.T.mean(1)
                else:
                    learner_explanations[learner][global_att] = vim.T

        # Precompute teacher targets once — reused across all explainers and k
        tau_tr = _predict_teacher(model, x_train, "CATENets")
        tau_te = _predict_teacher(model, x_test, "CATENets")

        # Compute zero baseline once
        baseline_eval = np.zeros(baseline.shape) if zero_baseline else baseline

        # Calculate IF-PEHE for insertion and deletion for each explanation methods

        for explainer_name in explainers:

            train_score_results = []
            train_mse_results = []
            test_score_results = []
            test_mse_results = []

            # obtaining global & local ranking for insertion & deletion

            if explainer_name == "permucate":
                exp = learner_explanations[learner][explainer_name]
                local_rank = attribution_ranking(exp)
                global_rank = np.flip(np.argsort(np.abs(exp.mean(0))))
            else:
                abs_explanation = np.abs(learner_explanations[learner][explainer_name])
                local_rank = attribution_ranking(
                    learner_explanations[learner][explainer_name]
                )
                global_rank = np.flip(np.argsort(abs_explanation.mean(0)))

            print("Calculating insertion/deletion results. ")
            insertion_results, deletion_results = insertion_deletion(
                (x_test[:explainer_limit], w_test[:explainer_limit], y_test[:explainer_limit]),
                local_rank,
                model,
                baseline_eval,
                selection_types,
                nuisance_functions,
            )

            for feature_idx in range(1, feature_size + 1):

                print(
                    "obtaining subgroup results for %s, feature_num: %s."
                    % (explainer_name, feature_idx),
                    end="\r",
                )

                # Starting from 1 features
                train_score, test_score, train_mse, test_mse = qini_score(
                    global_rank[:feature_idx],
                    (x_train, w_train, y_train),
                    (x_test, w_test, y_test),
                    model,
                    "CATENets",
                    tau_tr=tau_tr,
                    tau_te=tau_te,
                )

                train_score_results.append(train_score)
                test_score_results.append(test_score)
                test_mse_results.append(test_mse)
                train_mse_results.append(train_mse)

            insertion_deletion_data.append(
                [
                    learner,
                    explainer_name,
                    insertion_results,
                    deletion_results,
                    train_score_results,
                    train_mse_results,
                    test_score_results,
                    test_mse_results,
                    [qini_score_cal(w_train, y_train, results_train[i])],
                    [qini_score_cal(w_test, y_test, results_test[i])],
                ]
            )

        with open(
            os.path.join(
                f"results/{cohort_name}/",
                (
                    "insertion_deletion_shuffle_"
                    f"{shuffle}_{learner}_"
                    f"zero_baseline_{zero_baseline}_"
                    f"seed_{i}.pkl"
                ),
            ),
            "wb",
        ) as output_file:
            pickle.dump(insertion_deletion_data, output_file)

    ensemble_distillation_data = []
    ensemble_teacher = EnsembleTeacher(teachers, model_type=learner)

    # Precompute ensemble teacher targets once
    tau_tr_ens = ensemble_teacher.predict(x_train)
    tau_te_ens = ensemble_teacher.predict(x_test)

    for m in explainers:
        # 1) per-method global ranking
        A = learner_explanations[learner][m]
        # shape (n, d) for this method on this trial
        global_rank_m = np.flip(np.argsort(np.abs(A).mean(0)))

        # 2) distill to ensemble teacher using THIS method's rank
        m_train_score, m_test_score = [], []
        m_train_mse, m_test_mse = [], []

        for k in tqdm(range(1, feature_size + 1)):
            print(f"Computing distillation loss for ensemble with feature {k}")
            tr_sc, te_sc, tr_mse, te_mse = qini_score(
                global_rank_m[:k],
                (x_train, w_train, y_train),
                (x_test, w_test, y_test),
                teacher=ensemble_teacher,
                model_type="CATENets",
                tau_tr=tau_tr_ens,
                tau_te=tau_te_ens,
            )
            m_train_score.append(tr_sc)
            m_test_score.append(te_sc)
            m_train_mse.append(tr_mse)
            m_test_mse.append(te_mse)

        ensemble_distillation_data.append(
            [
                "EnsembleTeacherOverTrials",  # learner
                f"{m}|ensemble_teacher",  # method_name (tag with ensemble)
                m_train_score,  # train_score_results
                m_train_mse,  # train_mse_results
                m_test_score,  # test_score_results
                m_test_mse,  # test_mse_results
                [qini_score_cal(w_train, y_train, ensemble_teacher.predict(x_train))],
                [qini_score_cal(w_test, y_test, ensemble_teacher.predict(x_test))],
            ]
        )

    with open(
        os.path.join(
            f"results/{cohort_name}/",
            (
                f"ensemble_shuffle_{shuffle}_{learner}_"
                f"zero_baseline_{zero_baseline}.pkl"
            ),
        ),
        "wb",
    ) as output_file:
        pickle.dump(ensemble_distillation_data, output_file)

    with open(
        os.path.join(
            f"results/{cohort_name}/",
            f"train_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}.pkl",
        ),
        "wb",
    ) as output_file:
        pickle.dump(results_train, output_file)

    with open(
        os.path.join(
            f"results/{cohort_name}",
            f"test_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}.pkl",
        ),
        "wb",
    ) as output_file:
        pickle.dump(results_test, output_file)