"""Model related utility functions"""
import numpy as np
import torch
import xgboost as xgb
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import src.CATENets.catenets.models as cate_models
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.permucate.learners import CateNet, CausalForest, DRLearner


class TwoLayerMLP(nn.Module):
    """Simple two-layer MLP model"""

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def train_model(
        self,
        x,
        y,
        epochs=100,
        lr=1e-4,
        batch_size=32,
        validation_split=0.1,
        patience=10,
    ):
        # Convert data into DataLoader for mini-batch processing
        dataset = TensorDataset(
            torch.from_numpy(x).float(), torch.from_numpy(y).float()
        )

        # Splitting data into training and validation sets
        train_len = int((1 - validation_split) * len(dataset))
        valid_len = len(dataset) - train_len
        train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_valid_loss = float("inf")
        counter = 0

        for epoch in range(epochs):
            # Training loop
            for batch_x, batch_y in train_loader:
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loop
            with torch.no_grad():
                valid_loss = sum(
                    criterion(self(batch_x), batch_y)
                    for batch_x, batch_y in valid_loader
                )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                counter = 0  # Reset counter when new best model is found
            else:
                counter += 1  # Increment counter when no improvement in validation loss

                if counter >= patience:
                    print("Early stopping!")
                    return

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}",
                    f"Validation Loss: {valid_loss:.4f}",
                )


class NuisanceFunctions:
    """Nuisance functions for propensity scores."""

    def __init__(self, rct: bool):

        self.rct = rct
        self.mu0 = xgb.XGBClassifier()
        self.mu1 = xgb.XGBClassifier()

        self.m = xgb.XGBClassifier()

        self.rf = xgb.XGBClassifier(
            # reg_lambda=2,
            # max_depth=3,
            # colsample_bytree=0.2,
            # min_split_loss=10
        )
        # self.rf = LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])

    def fit(self, x_val, y_val, w_val):

        x0, x1 = x_val[w_val == 0], x_val[w_val == 1]
        y0, y1 = y_val[w_val == 0], y_val[w_val == 1]

        self.mu0.fit(x0, y0)
        self.mu1.fit(x1, y1)
        self.m.fit(x_val, y_val)
        self.rf.fit(x_val, w_val)

    def predict_mu_0(self, x):
        return self.mu0.predict(x)

    def predict_mu_1(self, x):
        return self.mu1.predict(x)

    def predict_propensity(self, x):
        if self.rct:
            p = 0.5 * np.ones(x.shape[0])
        else:
            p = self.rf.predict_proba(x)[:, 1]
        return p

    def predict_m(self, x):
        return self.m.predict(x)


def init_model(x_train, y_train, model_type, device):
    """Initialize new CATE model"""
    models = {
        "XLearner": pseudo_outcome_nets.XLearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        ),
        "SLearner": cate_models.torch.SLearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        ),
        "RLearner": pseudo_outcome_nets.RLearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            n_iter=1000,
            lr=1e-3,
            patience=10,
            batch_size=128,
            batch_norm=False,
            nonlin="relu",
            device=device,
        ),
        "RALearner": pseudo_outcome_nets.RALearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            n_iter=1000,
            lr=1e-3,
            patience=10,
            batch_size=128,
            batch_norm=False,
            nonlin="relu",
            device=device,
        ),
        "TLearner": cate_models.torch.TLearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        ),
        "TARNet": cate_models.torch.TARNet(
            x_train.shape[1],
            binary_y=True,
            n_layers_r=1,
            n_layers_out=1,
            n_units_out=100,
            n_units_r=100,
            batch_size=128,
            n_iter=1000,
            batch_norm=False,
            early_stopping=True,
            nonlin="relu",
        ),
        "CFRNet_0.01": cate_models.torch.TARNet(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_r=2,
            n_layers_out=2,
            n_units_out=100,
            n_units_r=100,
            batch_size=128,
            n_iter=1000,
            lr=1e-3,
            batch_norm=False,
            nonlin="relu",
            penalty_disc=0.01,
        ),
        "CFRNet_0.001": cate_models.torch.TARNet(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_r=2,
            n_layers_out=2,
            n_units_out=100,
            n_units_r=100,
            lr=1e-5,
            batch_size=128,
            n_iter=1000,
            batch_norm=False,
            nonlin="relu",
            penalty_disc=0.001,
        ),
        "DRLearner": pseudo_outcome_nets.DRLearner(
            x_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        ),
        "CausalForest": CausalForest()

    }

    return models[model_type]
