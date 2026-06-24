import numpy as np


def compute_r_risk(
    df_test,
    x_cols,
    cate_estimator,
    df_train=None,
    m_hat=None,
    pi_hat=None,
    model_m=None,
    model_pi=None,
    **kwargs,
):
    """
    Compute the feasible r-risk of the CATE estimator.
    r_risk = ((Y - m) - (A - pi) * tau_hat)^2
    This risk is feasible when estimates of m and pi are used.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data
    x_cols : list
        List of covariates columns
    cate_estimator : object
        CATE estimator, must be fitted
    df_train : pd.DataFrame, optional
        Training data, by default None
    m_hat : np.ndarray, optional
        Estimated mean outcome, by default None
    pi_hat : np.ndarray, optional
        Estimated propensity score, by default None
    model_m : object, optional
        Model for the mean outcome, by default None
    model_pi : object, optional
        Model for the propensity score, by default None
    """
    if m_hat is None:
        print("fit m_hat...")
        model_m.fit(
            X=df_train[x_cols],
            y=df_train["y"],
        )
        m_hat = model_m.predict(df_test[x_cols])
    if pi_hat is None:
        print("fit pi_hat...")
        model_pi.fit(
            X=df_train[x_cols],
            y=df_train["a"],
        )
        pi_hat = model_pi.predict_proba(df_test[x_cols])[
            :, np.argwhere(model_pi.classes_ == 1.0).ravel()[0]
        ]

    tau_hat = cate_estimator.predict(df_test[x_cols].values).detach().cpu().numpy().flatten()
    r_risk = np.array((df_test["y"] - m_hat) - (df_test["a"] - pi_hat) * tau_hat) ** 2

    return r_risk


def compute_tau_risk(
    df_test, x_cols, cate_estimator, tau_true=None, df_train=None, **kwargs
):
    """
    Computes the oracle tau-risk of the CATE estimator, corresponding to the
    squared error between the true CATE and the estimated CATE.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data
    x_cols : list
        List of covariates columns
    cate_estimator : object
        CATE estimator, must be fitted
    df_train : pd.DataFrame, optional
        For consistency with other functions, by default None
    tau_true : np.ndarray, optional
        True CATE, by default None
    """

    if tau_true is None:
        tau_true = df_test["tau"]

    tau_hat = cate_estimator.predict(df_test[x_cols].values).detach().cpu().numpy().flatten()
    tau_risk = np.array(tau_true - tau_hat) ** 2
    return tau_risk


def compute_pseudo_outcome_risk(
    df_test,
    x_cols,
    cate_estimator,
    df_train=None,
    pi_hat=None,
    mu_1_hat=None,
    mu_0_hat=None,
    model_mu=None,
    model_pi=None,
    pseudo_outcomes=None,
    **kwargs,
):
    """
    Compute the pseudo-outcome risk of the CATE estimator. This corresponds to
    the squared error between the pseudo-outcomes and the estimated CATE.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data
    x_cols : list
        List of covariates columns
    cate_estimator : object
        CATE estimator, must be fitted
    df_train : pd.DataFrame, optional
        Training data, by default None
    pi_hat : np.ndarray, optional
        Estimated propensity score, by default None
    mu_1_hat : np.ndarray, optional
        Estimated response for treated unit, by default None
    mu_0_hat : np.ndarray, optional
        Estimated response for control unit, by default None
    model_mu : object, optional
        Model for the response, by default None
    model_pi : object, optional
        Model for the propensity score, by default None
    pseudo_outcomes : np.ndarray, optional
        Pseudo-outcomes, by default None
    """
    if pseudo_outcomes is None:
        if mu_1_hat is None:
            model_mu.fit(
                X=df_train[x_cols],
                y=df_train["y"] * df_train["a"],
            )
            mu_1_hat = model_mu.predict(df_test[x_cols])
        if mu_0_hat is None:
            model_mu.fit(
                X=df_train[x_cols],
                y=df_train["y"] * (1 - df_train["a"]),
            )
            mu_0_hat = model_mu.predict(df_test[x_cols])
        if pi_hat is None:
            model_pi.fit(
                X=df_train[x_cols],
                y=df_train["a"],
            )
            pi_hat = model_pi.predict_proba(df_test[x_cols])[
                :, np.argwhere(model_pi.classes_ == 1.0).ravel()[0]
            ]

        mu_a = df_test["a"] * mu_1_hat + (1 - df_test["a"]) * mu_0_hat
        pseudo_outcomes = (
            (df_test["y"] - mu_a) * (df_test["a"] - pi_hat) / (pi_hat * (1 - pi_hat))
            + mu_1_hat
            - mu_0_hat
        )

    risk = (
        np.array(pseudo_outcomes - cate_estimator.predict(df_test[x_cols].values)) ** 2
    )
    return risk