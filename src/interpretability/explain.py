"""Module for explainability methods."""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import (
    DeepLift,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
    Saliency,
    ShapleyValueSampling,
)
from captum.attr._core.lime import get_exp_kernel_similarity_function
from sklift.metrics import qini_auc_score
from torch import nn

# import shap


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _SqueezeWrapper(nn.Module):
    """Wraps a model that outputs [N, 1] to output [N] for captum compatibility."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # Force eval mode so BatchNorm works with any batch size.
        # Do NOT use torch.no_grad() — gradient-based methods need grads.
        was_training = self.model.training
        self.model.eval()
        out = self.model(*args, **kwargs)
        if was_training:
            self.model.train()
        if out.dim() > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class Explainer:
    """Explainer instance."""

    def __init__(
        self,
        model: nn.Module,
        feature_names: List,
        explainer_list: List = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "deeplift",
            "shapley_value_sampling",
            "lime",
        ],
        n_steps: int = 500,
        perturbations_per_eval: int = 50,
        n_samples: int = 2500,
        n_baselines: int = 25,
        kernel_width: float = 1.0,
        baseline: Optional[torch.Tensor] = None,
        x_train: Optional[np.ndarray] = None,
    ) -> None:

        # Infer device from model parameters; fall back to CPU
        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cpu")
        self.baseline = baseline
        self.x_train = x_train
        self.explainer_list = explainer_list
        self.feature_names = feature_names

        # Wrap model so output is [N] instead of [N, 1] for captum
        model = _SqueezeWrapper(model)

        # Feature ablation
        feature_ablation_model = FeatureAblation(model)

        def feature_ablation_cbk(x_test: torch.Tensor) -> torch.Tensor:
            out = feature_ablation_model.attribute(
                x_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

            return out

        # Integrated gradients
        integrated_gradients_model = IntegratedGradients(model)

        def integrated_gradients_cbk(x_test: torch.Tensor) -> torch.Tensor:
            test_values = torch.zeros_like(x_test)
            batch_size = 32
            for i in range(0, len(x_test), batch_size):
                end_idx = min(i + batch_size, len(x_test))
                test_values[i:end_idx] = integrated_gradients_model.attribute(
                    x_test[i:end_idx],
                    n_steps=n_steps,
                )

            return test_values

        def baseline_integrated_gradients_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = torch.zeros_like(x_test)
            batch_size = 32
            for i in range(0, len(x_test), batch_size):
                end_idx = min(i + batch_size, len(x_test))
                test_values[i:end_idx] = integrated_gradients_model.attribute(
                    x_test[i:end_idx], n_steps=n_steps, baselines=self.baseline
                )
            return test_values

        def smooth_grad_cpk(x_test: torch.Tensor) -> torch.Tensor:

            noise_tunnel = NoiseTunnel(integrated_gradients_model)
            test_values = torch.zeros_like(x_test)
            batch_size = 32
            for i in range(0, len(x_test), batch_size):
                end_idx = min(i + batch_size, len(x_test))
                test_values[i:end_idx] = noise_tunnel.attribute(
                    (x_test[i:end_idx]), nt_type="smoothgrad_sq"
                )
            return test_values

        # DeepLift
        deeplift_model = DeepLift(model)

        def deeplift_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return deeplift_model.attribute(x_test)

        # Feature permutation
        feature_permutation_model = FeaturePermutation(model)

        def feature_permutation_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return feature_permutation_model.attribute(
                x_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

        # LIME
        exp_eucl_distance = get_exp_kernel_similarity_function(
            kernel_width=kernel_width
        )

        lime_model = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=exp_eucl_distance,
        )

        def lime_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = torch.zeros((x_test.size()))

            for test_ind in range(len(x_test)):

                lime_value = lime_model.attribute(
                    x_test[test_ind].view(1, -1),
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                )

                test_values[test_ind] = lime_value.detach().cpu()

            return self._check_tensor(test_values)

        def baseline_lime_cbk(x_test: torch.Tensor) -> torch.Tensor:

            test_values = torch.zeros((x_test.size()))

            for test_ind in range(len(x_test)):

                lime_value = lime_model.attribute(
                    x_test[test_ind].view(1, -1),
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    baselines=self.baseline,
                )

                test_values[test_ind] = lime_value.detach().cpu()

            return self._check_tensor(test_values)

        # Baseline shapley value sampling
        def qini_score_wrapper(x_test):
            with torch.no_grad():
                x_hat = model.predict(x_test).flatten().detach().cpu().numpy()
                score = qini_auc_score(self.y_test, x_hat, self.w_test)

                score = self._check_tensor(score)
                return score  # pragma: no cover – unused legacy code

        # Initialize Shapley Value Sampling model
        # shapley_value_sampling_model = ShapleyValueSampling(qini_score_wrapper)

        shapley_value_sampling_model = ShapleyValueSampling(model)

        def baseline_shapley_value_sampling_cbk(
            x_test: torch.Tensor,
        ) -> torch.Tensor:

            return shapley_value_sampling_model.attribute(
                x_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=True,
            )

        # Marginal shapley value sampling — average over n_baselines

        def marginal_shapley_value_sampling_cbk(x_test: torch.Tensor) -> torch.Tensor:
            if self.x_train is None:
                # Fall back to single zero-baseline call
                return shapley_value_sampling_model.attribute(
                    x_test,
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    baselines=self.baseline,
                    show_progress=True,
                )

            # Average attributions over n_baselines sampled from training data
            n_test = x_test.shape[0]
            accum = torch.zeros_like(x_test)
            for b in range(n_baselines):
                indices = np.random.choice(len(self.x_train), size=n_test, replace=True)
                train_baselines = self._check_tensor(
                    torch.from_numpy(np.asarray(self.x_train[indices])).float()
                )
                attr = shapley_value_sampling_model.attribute(
                    x_test,
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    baselines=train_baselines,
                    show_progress=True,
                )
                accum += attr
            return accum / n_baselines

        # Kernel SHAP
        kernel_shap_model = KernelShap(model)

        def kernel_shap_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return kernel_shap_model.attribute(
                x_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                baselines=self.baseline,
                show_progress=True,
            )

        # Gradient SHAP
        gradient_shap_model = GradientShap(model)

        def gradient_shap_cbk(x_test: torch.Tensor) -> torch.Tensor:
            return gradient_shap_model.attribute(x_test, baselines=self.baseline)

        saliency_model = Saliency(model)

        def saliency_cpk(x_test: torch.tensor) -> torch.Tensor:
            test_values = torch.zeros_like(x_test)

            batch_size = 32
            for i in range(0, len(x_test), batch_size):
                end_idx = min(i + batch_size, len(x_test))
                test_values[i:end_idx] = saliency_model.attribute(x_test[i:end_idx])

            return self._check_tensor(test_values)

        def dummy_cbk(x_test: torch.Tensor) -> torch.Tensor:
            """Dummy function, returns zero tensor"""
            return torch.zeros((x_test.size()))

        def random_cbk(x_test: torch.Tensor) -> torch.Tensor:
            """Baseline function, returns random ranking"""
            n_samples, n_features = x_test.shape

            ranks = torch.empty((n_samples, n_features), dtype=torch.long)
            base = torch.arange(n_features)

            for i in range(n_samples):
                ranks[i] = base[torch.randperm(n_features)]

            return ranks

        self.explainers = {
            "feature_ablation": feature_ablation_cbk,
            "integrated_gradients": integrated_gradients_cbk,
            "baseline_integrated_gradients": baseline_integrated_gradients_cbk,
            "smooth_grad": smooth_grad_cpk,
            "deeplift": deeplift_cbk,
            "feature_permutation": feature_permutation_cbk,
            "lime": lime_cbk,
            "baseline_lime": baseline_lime_cbk,
            "baseline_shapley_value_sampling": baseline_shapley_value_sampling_cbk,
            "marginal_shapley_value_sampling": marginal_shapley_value_sampling_cbk,
            "kernel_shap": kernel_shap_cbk,
            "gradient_shap": gradient_shap_cbk,
            "saliency": saliency_cpk,
            "loco": dummy_cbk,
            "permucate": dummy_cbk,
            "random": random_cbk,
        }

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(self.device)

    def explain(self, X: torch.Tensor, W: torch.Tensor = None, Y: torch.Tensor = None) -> Dict:
        output = {}

        x_test = self._check_tensor(X)
        x_test.requires_grad_()

        if self.baseline is None:
            self.baseline = torch.zeros(
                x_test.shape, device=self.device
            )  # Zero tensor as baseline if no baseline specified
        else:
            self.baseline = self._check_tensor(self.baseline)

        self.w_test = W
        self.y_test = Y

        for name in self.explainer_list:

            explainer = self.explainers[name]
            output[name] = explainer(x_test).detach().cpu().numpy()
        return output

    def plot(self, X: torch.Tensor) -> None:
        explanations = self.explain(X)

        fig, axs = plt.subplots(int((len(explanations) + 1) / 2), 2)

        idx = 0
        for name in explanations:
            x_pos = np.arange(len(self.feature_names))

            ax = axs[int(idx / 2), idx % 2]

            ax.bar(x_pos, np.mean(np.abs(explanations[name]), axis=0), align="center")
            ax.set_xlabel("Features")
            ax.set_title(f"{name}")

            idx += 1
        plt.tight_layout()
