"""Dataset classes."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils import InvertableColumnTransformer


class Dataset:
    """Data wrapper for clinical data."""

    def __init__(self, cohort_name, random_state=42, shuffle=False):

        self.cohort_name = cohort_name
        self.shuffle = shuffle
        self.random_state = random_state

        self._load_data(cohort_name)  # Create self.data
        self._split_data(self.data.copy())

    def _load_data(self, cohort_name):
        """Load data"""
        if cohort_name in ["massive_trans", "responder"]:
            self.data = self._load_pickle_data(cohort_name)
        elif cohort_name == "ist3":
            self.data = self._load_ist3_data()
        elif cohort_name == "crash_2":
            self.data = self._load_crash2_data()
        elif cohort_name == "txa":
            self.data = self._load_txa_data()
        elif cohort_name == "sprint":
            self.data = self._load_sprint_data()
        elif cohort_name == "accord":
            self.data = self._load_accord_data()
        elif cohort_name == "accord_glycemia":
            self.data = self._load_accord_glycemia_data()
        elif cohort_name == "sprint_filter":
            self.data = self._load_sprint_filter_data()
        elif cohort_name == "accord_filter":
            self.data = self._load_accord_filter_data()
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")

        self.continuous_indices = [
            self.data.columns.get_loc(col) for col in self.continuous_vars
        ]

        self.categorical_indices = self.get_one_hot_column_indices(
            self.data.drop([self.treatment, self.outcome], axis=1),
            self.categorical_vars,
        )

        self.discrete_indices = self.get_one_hot_column_indices(
            self.data.drop([self.treatment, self.outcome], axis=1),
            self.categorical_vars + self.binary_vars,
        )

    def _load_pickle_data(self, cohort_name):

        if cohort_name == "responder":
            data = pd.read_pickle("data/trauma_responder.pkl")
        elif cohort_name == "massive_trans":
            data = pd.read_pickle("data/low_bp_survival.pkl")
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")

        filter_regex = [
            "proc",
            "ethnicity",
            "residencestate",
            "toxicologyresults",
            "registryid",
            "COV",
            "TT",
            "scenegcsmotor",
            "scenegcseye",
            "scenegcsverbal",
            "edgcsmotor",
            "edgcseye",
            "edgcsverbal",
            "sex_F",
            "traumatype_P",
            "traumatype_OTHER",
            "causecode",
        ]

        self.treatment = "treated"
        self.outcome = "outcome"

        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        self.binary_vars = [
            "sex_M",
            "traumatype_B",
        ]

        self.continuous_vars = [
            "age",
            "scenegcs",
            "scenefirstbloodpressure",
            "scenefirstpulse",
            "scenefirstrespirationrate",
            "edfirstbp",
            "edfirstpulse",
            "edfirstrespirationrate",
            "edgcs",
            "temps2",
            "BD",
            "CFSS",
            "COHB",
            "CREAT",
            "FIB",
            "FIO2",
            "HCT",
            "HGB",
            "INR",
            "LAC",
            "NA",
            "PAO2",
            "PH",
            "PLTS",
        ]

        # self.categorical_vars = [col for col in data.columns if 'causecode' in col]
        self.categorical_vars = []

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        return data

    def _load_txa_data(self):

        data = pd.read_pickle("data/txa_cohort.pkl")

        filter_regex = [
            "proc",
            "ethnicity",
            "residencestate",
            "toxicologyresults",
            "registryid",
            "COV",
            "TT",
            "scenegcsmotor",
            "scenegcseye",
            "scenegcsverbal",
            "edgcsmotor",
            "edgcseye",
            "edgcsverbal",
            "sex_F",
            "traumatype_P",
            "traumatype_OTHER",
            "causecode",
        ]

        self.treatment = "treated"
        self.outcome = "outcome"

        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        self.binary_vars = [
            "sex_M",
            "traumatype_B",
        ]

        self.continuous_vars = [
            "age",
            "scenefirstbloodpressure",
            "scenefirstpulse",
            "scenefirstrespirationrate",
            "scenegcs",
        ]

        self.categorical_vars = []
        # For continuous variables

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]
        imp_mean = SimpleImputer(strategy="mean")
        data[self.continuous_vars] = imp_mean.fit_transform(data[self.continuous_vars])
        # Instantiate the Matcher class

        X = data.drop(columns=[self.treatment, self.outcome])
        y = data[self.treatment]

        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        data["propensity_score"] = model.predict_proba(X)[:, 1]

        treated = data[data[self.treatment] == 1].copy()
        control = data[data[self.treatment] == 0].copy()

        # Fit the nearest neighbors model for 1 neighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity_score"]])

        # Find the nearest neighbor indices for the treated group
        _, indices = nbrs.kneighbors(treated[["propensity_score"]])

        # Flatten indices for 1:2 matching & Extract matched controls

        matched_control = control.iloc[indices.flatten()]
        matched_data = pd.concat([treated, matched_control]).sort_index()
        matched_data = matched_data.sort_index()
        matched_data.drop(columns=["propensity_score"], inplace=True)

        return matched_data

    def _load_ist3_data(self):
        data = pd.read_sas("data/datashare_aug2015.sas7bdat")

        self.outcome = "aliveind6"
        self.treatment = "itt_treat"

        self.continuous_vars = [
            "age",
            "weight",
            "glucose",
            # "gcs_eye_rand",
            # "gcs_motor_rand",
            # "gcs_verbal_rand",
            "gcs_score_rand",
            "nihss",
            "sbprand",
            "dbprand",
        ]

        self.categorical_vars = [
            "stroketype",
        ]

        self.binary_vars = ["gender", "antiplat_rand", "atrialfib_rand", "infarct"]

        data = data[data.stroketype != 5]

        data["antiplat_rand"] = np.where(data["antiplat_rand"] == 1, 1, 0)
        data["atrialfib_rand"] = np.where(data["atrialfib_rand"] == 1, 1, 0)

        data["gender"] = np.where(data["gender"] == 2, 1, 0)

        data["infarct"] = np.where(data["infarct"] == 0, 0, 1)

        data[self.treatment] = np.where(data[self.treatment] == 0, 1, 0)
        data[self.outcome] = np.where(data[self.outcome] == 1, 1, 0)

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        # data = data.sample(2500)

        return data

    def _load_crash2_data(self):

        self.outcome = "outcome"
        self.treatment = "treatment_code"

        data = pd.read_excel("data/crash_2.xlsx")

        self.continuous_vars = [
            "iage",
            "isbp",
            "irr",
            "icc",
            "ihr",
            "ninjurytime",
            "igcs",
        ]
        self.categorical_vars = ["iinjurytype"]
        self.binary_vars = [
            "isex",
        ]

        data = data.drop(
            data[(data[self.treatment] == "P") | (data[self.treatment] == "D")].index
        )

        data = data[data.iinjurytype != 3]
        # data["iinjurytype"] = np.where(data["iinjurytype"]== 1, 0, 1)
        # Set blunt to 0

        data["isex"] = np.where(data["isex"] == 2, 0, 1)

        # deal with missing data

        data["irr"] = np.where(data["irr"] == 0, np.nan, data["irr"])
        data["icc"] = np.where(data["icc"] >= 20, np.nan, data["icc"])

        data["isbp"] = np.where(data["isbp"] == 999, np.nan, data["isbp"])
        data["isbp"] = np.where(data["isbp"] == 0, np.nan, data["isbp"])

        data["ninjurytime"] = np.where(
            data["ninjurytime"] >= 20, np.nan, data["ninjurytime"]
        )
        data["ninjurytime"] = np.where(
            data["ninjurytime"] == 0, np.nan, data["ninjurytime"]
        )
        # data["ninjurytime"] = np.where(
        #   data["ninjurytime"] == 999, np.nan, data["ninjurytime"]
        # )

        data[self.treatment] = np.where(data[self.treatment] == "Active", 1, 0)
        data[self.outcome] = np.where(data["icause"].isna(), 1, 0)

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        # data["iinjurytype_1"] = np.where(data["iinjurytype_2"]== 1, 0, 1)
        # data.pop("iinjurytype_2")

        # data.pop("iinjurytype_3")

        # data = data.sample(int(len(data)*0.95))

        return data

    def _load_sprint_data(self):

        self.outcome = "event_primary"
        self.treatment = "intensive"

        outcome = pd.read_csv("data/sprint/outcomes.csv")
        baseline = pd.read_csv("data/sprint/baseline.csv")

        baseline.columns = [x.lower() for x in baseline.columns]
        outcome.columns = [x.lower() for x in outcome.columns]

        data = baseline.merge(outcome, on="maskid", how="inner")

        # Ordinal encoding: never=0, former=1, current=2, missing(4)→NaN
        # Preserves the never < former < current gradient rather than collapsing never and former
        data["smoke_3cat"] = np.where(
            data["smoke_3cat"] == 4, np.nan,
            np.where(data["smoke_3cat"] == 3, 2,
            np.where(data["smoke_3cat"] == 2, 1, 0))
        )

        self.continuous_vars = [
            "age",
            "sbp",
            "dbp",
            # "n_agents",
            # "egfr",  # dropped: retrieves irrelevant nephrology papers; CKD signal captured by sub_ckd
            # "screat",
            # "ualb",  # not present in sprint baseline CSV
            "chr",
            "glur",
            "hdl",
            "trr",
            "umalcr",
            "bmi",
            "smoke_3cat",  # ordinal: never=0, former=1, current=2; treated as numeric for imputation/scaling
            # "risk10yrs"
        ]

        self.binary_vars = [
            "female",
            "race_black",
            "aspirin",
            "statin",
            "sub_cvd",
            "sub_ckd"
            # "inclusionfrs"
            # "noagents"
        ]

        self.categorical_vars = []

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data[self.outcome] = np.where(data[self.outcome] == 1, 0, 1)
        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _load_sprint_filter_data(self):

        self.outcome = "event_primary"
        self.treatment = "intensive"

        outcome = pd.read_csv("data/sprint/outcomes.csv")
        baseline = pd.read_csv("data/sprint/baseline.csv")

        baseline.columns = [x.lower() for x in baseline.columns]
        outcome.columns = [x.lower() for x in outcome.columns]

        data = baseline.merge(outcome, on="maskid", how="inner")

        data["smoke_3cat"] = np.where(
            data["smoke_3cat"] == 4, np.nan, np.where(data["smoke_3cat"] == 3, 1, 0)
        )

        self.continuous_vars = [
            "age",
            "sbp",
            "dbp",
            # "n_agents",
            "egfr",
            # "screat",
            "chr",
            "glur",
            "hdl",
            "trr",
            "umalcr",
            "bmi",
            # "risk10yrs"
        ]

        self.binary_vars = [
            "female",
            "race_black",
            "smoke_3cat",
            "aspirin",
            "statin",
            "sub_cvd",
            # "sub_ckd"
            # "inclusionfrs"
            # "noagents"
        ]

        self.categorical_vars = []

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data[self.outcome] = np.where(data[self.outcome] == 1, 0, 1)

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _load_accord_data(self):

        data = pd.read_csv("data/accord/accord.csv")

        # ── add baseline HbA1c ─────────────────────────────────────────
        hba = pd.read_csv("data/accord/hba1c.csv")
        hba_bl = hba[hba["Visit"] == "BLR"][["MaskID", "hba1c"]]
        data = data.merge(hba_bl, on="MaskID", how="inner")

        # ── add years of diabetes ──────────────────────────────────────
        phys = pd.read_csv("data/accord/baselinehistoryphysicalexam.csv")
        data = data.merge(phys[["MaskID", "yrsdiab"]], on="MaskID", how="inner")

        self.outcome = "censor_po"
        self.treatment = "treatment"

        self.continuous_vars = [
            "baseline_age",
            "bmi",
            "hba1c",
            "yrsdiab",
            "sbp",
            "dbp",
            "hr",
            "fpg",
            # "alt",
            # "cpk",
            # "potassium",
            # "screat",
            "gfr",
            # 'ualb',
            # 'ucreat',
            "uacr",
            # "chol",
            "trig",
            # "vldl",
            "ldl",
            "hdl",
            "bp_med",
        ]

        self.binary_vars = [
            "female",
            "raceclass",
            "cvd_hx_baseline",
            "statin",
            "aspirin",
            # "antiarrhythmic",
            # "anti_coag",
            # 'dm_med',
            # 'cv_med',
            # 'lipid_med',
            "x4smoke",
        ]

        self.categorical_vars = []

        data["treatment"] = np.where(
            data["treatment"].str.contains("Intensive BP", na=False), 1, 0
        )
        data["raceclass"] = np.where(data["raceclass"] == "Black", 1, 0)
        data["x4smoke"] = np.where(data["x4smoke"] == 1, 1, 0)

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _load_accord_glycemia_data(self):
        """Load ACCORD glycemic-control arm (all 10 251 participants).

        Treatment: Intensive Glycemia = 1, Standard Glycemia = 0.
        Baseline covariates are reconstructed from the raw ACCORD tables
        because the pre-built accord.csv only contains the BP-trial subset.
        """

        # ── treatment & demographics from the key file ──────────────────
        key = pd.read_csv("data/accord/accord_key.csv")
        key["treatment"] = np.where(
            key["treatment"].str.contains("Intensive G", na=False), 1, 0
        )
        key["raceclass"] = np.where(key["raceclass"] == "Black", 1, 0)

        # ── outcome ─────────────────────────────────────────────────────
        cvd = pd.read_csv("data/accord/cvdoutcomes.csv")

        # ── baseline HbA1c ─────────────────────────────────────────────
        hba = pd.read_csv("data/accord/hba1c.csv")
        hba_bl = hba[hba["Visit"] == "BLR"][["MaskID", "hba1c"]]

        # ── baseline physical exam (BMI, smoking, years of diabetes) ───
        phys = pd.read_csv("data/accord/baselinehistoryphysicalexam.csv")
        phys["bmi"] = phys["wt_kg"] / (phys["ht_cm"] / 100) ** 2
        # cigarett: 1 = current smoker, 2 = not current
        phys["x4smoke"] = np.where(phys["cigarett"] == 1, 1, 0)

        # ── baseline labs ──────────────────────────────────────────────
        labs = pd.read_csv("data/accord/otherlabs.csv")
        labs_bl = labs[labs["Visit"] == "BLR"].drop(columns=["Visit"])

        # ── baseline blood pressure ────────────────────────────────────
        bp = pd.read_csv("data/accord/bloodpressure.csv")
        bp_bl = bp[bp["Visit"] == "BLR"].drop(columns=["Visit"])

        # ── baseline lipids ────────────────────────────────────────────
        lip = pd.read_csv("data/accord/lipids.csv")
        lip_bl = lip[lip["Visit"] == "BLR"].drop(columns=["Visit"])

        # ── baseline medications ───────────────────────────────────────
        meds = pd.read_csv("data/accord/concomitantmeds.csv")
        meds_bl = meds[meds["Visit"] == "BLR"].drop(columns=["Visit"])
        bp_med_cols = [
            "loop", "thiazide", "ksparing", "a2rb", "acei",
            "dhp_ccb", "nondhp_ccb", "alpha_blocker", "central_agent",
            "beta_blocker", "vasodilator", "reserpine", "other_bpmed",
        ]
        meds_bl["bp_med"] = meds_bl[bp_med_cols].sum(axis=1)
        dm_med_cols = [
            "sulfonylurea", "biguanide", "meglitinide", "ag_inhibitor",
            "nphl_insulin", "tzd", "reg_insulin", "la_insulin",
            "othbol_insulin", "premix_insulin", "other_diabmed",
        ]
        meds_bl["dm_med"] = meds_bl[dm_med_cols].sum(axis=1)
        insulin_cols = [
            "nphl_insulin", "reg_insulin", "la_insulin",
            "othbol_insulin", "premix_insulin",
        ]
        meds_bl["insulin"] = (meds_bl[insulin_cols].sum(axis=1) > 0).astype(int)

        # ── merge everything on MaskID ─────────────────────────────────
        data = (
            key[["MaskID", "treatment", "female", "baseline_age",
                 "raceclass", "cvd_hx_baseline"]]
            .merge(cvd[["MaskID", "censor_po"]], on="MaskID", how="inner")
            .merge(hba_bl, on="MaskID", how="inner")
            .merge(phys[["MaskID", "bmi", "x4smoke", "yrsdiab"]], on="MaskID", how="inner")
            .merge(bp_bl, on="MaskID", how="inner")
            .merge(labs_bl, on="MaskID", how="inner")
            .merge(lip_bl[["MaskID", "trig", "ldl", "hdl"]], on="MaskID", how="inner")
            .merge(meds_bl[["MaskID", "bp_med", "dm_med", "insulin", "statin",
                           "aspirin", "antiarrhythmic", "anti_coag"]],
                   on="MaskID", how="inner")
        )

        self.outcome = "censor_po"
        self.treatment = "treatment"

        self.continuous_vars = [
            "baseline_age",
            "bmi",
            "hba1c",
            "yrsdiab",
            "sbp",
            "dbp",
            "hr",
            "fpg",
            "alt",
            "cpk",
            "potassium",
            "gfr",
            "uacr",
            "trig",
            "ldl",
            "hdl",
            "bp_med",
            "dm_med",
        ]

        self.binary_vars = [
            "female",
            "raceclass",
            "cvd_hx_baseline",
            "insulin",
            "statin",
            "aspirin",
            "antiarrhythmic",
            "anti_coag",
            "x4smoke",
        ]

        self.categorical_vars = []

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _load_accord_filter_data(self):

        data = pd.read_csv("data/accord/accord.csv")

        self.outcome = "censor_po"
        self.treatment = "treatment"

        self.continuous_vars = [
            "baseline_age",
            "sbp",
            "dbp",
            "bp_med",
            "gfr",
            "screat",
            "chol",
            "fpg",
            "hdl",
            "trig",
            "uacr",
            "bmi",
            # 'hr',
            # 'alt',
            # 'cpk',
            # 'potassium',
            # 'ualb',
            # 'ucreat',
            # 'vldl',
            # 'ldl',
        ]

        self.binary_vars = [
            "female",
            "raceclass",
            "x4smoke",
            "aspirin",
            "statin",
            "cvd_hx_baseline"
            # 'antiarrhythmic',
            # 'anti_coag',
            # 'dm_med',
            # 'cv_med',
            # 'lipid_med',
        ]

        self.categorical_vars = []

        data["treatment"] = np.where(
            data["treatment"].str.contains("Intensive BP", na=False), 1, 0
        )
        data["raceclass"] = np.where(data["raceclass"] == "Black", 1, 0)
        data["x4smoke"] = np.where(data["x4smoke"] == 1, 1, 0)

        data = data[
            self.continuous_vars
            + self.categorical_vars
            + self.binary_vars
            + [self.treatment]
            + [self.outcome]
        ]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _normalize_data(self, x: np.ndarray, type: str):

        if type == "minmax":
            self.scaler = MinMaxScaler()

        elif type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(x.values)
        x = self.scaler.transform(x.values)

        return x

    def _split_data(self, df_raw):

        t_col = self.treatment
        y_col = self.outcome

        # Drop rows where non-continuous columns have NaN (cannot be imputed).
        # Continuous NaNs are handled below by the train-fitted SimpleImputer.
        non_cont_cols = [c for c in df_raw.columns if c not in self.continuous_vars]
        mask = df_raw[non_cont_cols].notna().all(axis=1)
        df_raw = df_raw.loc[mask].reset_index(drop=True)

        # build X (features only), W, Y (as arrays)
        X_df = df_raw.drop(columns=[t_col, y_col])
        W_all = df_raw[t_col].values.astype(int)
        Y_all = df_raw[y_col].values.astype(int)

        if self.shuffle:
            rs = self.random_state
        else:
            rs = 42

        idx = np.arange(len(df_raw))
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.2, random_state=rs, stratify=W_all
        )
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=0.2, random_state=rs, stratify=W_all[tr_idx]
        )

        # slice dataframes
        X_tr_df, X_va_df, X_te_df = (
            X_df.iloc[tr_idx],
            X_df.iloc[va_idx],
            X_df.iloc[te_idx],
        )
        W_tr, W_va, W_te = W_all[tr_idx], W_all[va_idx], W_all[te_idx]
        Y_tr, Y_va, Y_te = Y_all[tr_idx], Y_all[va_idx], Y_all[te_idx]

        # fit imputers/scaler on TRAIN continuous columns only
        cont_idx_in_X = [X_df.columns.get_loc(c) for c in self.continuous_vars]

        imp_cont = SimpleImputer(strategy="mean").fit(X_tr_df.iloc[:, cont_idx_in_X])
        scl_cont = StandardScaler().fit(
            imp_cont.transform(X_tr_df.iloc[:, cont_idx_in_X])
        )

        self.scaler = scl_cont
        self._imp_cont_ = imp_cont
        self.continuous_indices = cont_idx_in_X  # used by get_unnorm_value()

        # helper: transform continuous slice, leave others untouched
        def transform_split(X_block_df):
            Xb = X_block_df.values.copy()
            Xb_cont = scl_cont.transform(
                imp_cont.transform(X_block_df.iloc[:, cont_idx_in_X])
            )
            Xb[:, cont_idx_in_X] = Xb_cont
            return Xb

        X_tr = transform_split(X_tr_df)
        X_va = transform_split(X_va_df)
        X_te = transform_split(X_te_df)

        # save per-split matrices (features only)
        self.x_train, self.x_val, self.x_test = X_tr, X_va, X_te
        self.w_train, self.w_val, self.w_test = W_tr, W_va, W_te
        self.y_train, self.y_val, self.y_test = Y_tr, Y_va, Y_te

        self.x = np.vstack([self.x_train, self.x_val, self.x_test])
        self.w = np.concatenate([self.w_train, self.w_val, self.w_test])
        self.y = np.concatenate([self.y_train, self.y_val, self.y_test])

    def get_data(self, set: str = None):

        if set == "train":
            return self.x_train, self.w_train, self.y_train
        elif set == "val":
            return self.x_val, self.w_val, self.y_val
        elif set == "test":
            return self.x_test, self.w_test, self.y_test
        else:
            return self.x, self.w, self.y

    def get_feature_range(self, feature: int) -> np.ndarray:
        """Return value range for a feature"""

        x_train_cont = self.scaler.inverse_transform(
            self.x_train[:, self.continuous_indices]
        )
        return float(x_train_cont[:, feature].max() - x_train_cont[:, feature].min())

    def get_unnorm_value(self, x: np.ndarray) -> np.ndarray:
        """Returns the inverse-transformed values of all features."""
        # Create a copy of x to avoid modifying the original array

        x_copy = x.copy()
        # Apply inverse transform only on continuous variables
        x_copy[:, self.continuous_indices] = self.scaler.inverse_transform(
            x_copy[:, self.continuous_indices]
        )
        # Exclude treatment and outcome indices
        return x_copy[:]

    def get_norm(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Get normalized values"""
        return self.scaler.transform(x[:, self.continuous_indices])

    def get_feature_names(self):
        """Return feature names"""
        return self.data.drop([self.treatment, self.outcome], axis=1).columns

    def get_cohort_name(self):
        """Return cohort name"""
        return self.cohort_name

    def get_one_hot_column_indices(self, df, prefixes):
        """Get the indices for one-hot encoded columns."""
        indices_dict = {}

        for prefix in prefixes:
            # Filter for one-hot encoded columns with the given prefix
            one_hot_cols = [col for col in df.columns if col.startswith(prefix)]

            # Get the indices for these columns
            indices_dict[prefix] = [df.columns.get_loc(col) for col in one_hot_cols]

        return indices_dict


def obtain_txa_baselines(unnorm=False) -> np.ndarray:
    """Obtain baselines for TXA"""
    crash2 = pd.read_excel("data/crash_2.xlsx")

    outcome = "outcome"
    treatment = "treatment_code"

    crash2_continuous_vars = ["iage", "isbp", "irr", "ihr", "igcs", "ninjurytime"]

    crash2_binary_vars = ["iinjurytype_1", "iinjurytype_2", "isex"]

    crash2 = crash2.drop(
        crash2[(crash2[treatment] == "P") | (crash2[treatment] == "D")].index
    )

    # Set blunt to 1 and drop Blunt and penetrating.
    crash2 = crash2[crash2.iinjurytype != 3]
    # crash2["iinjurytype"] = np.where(crash2["iinjurytype"] == 1, 1, 0)

    crash2["iinjurytype_1"] = np.where(crash2["iinjurytype"] == 1, 1, 0)
    crash2["iinjurytype_2"] = np.where(crash2["iinjurytype_1"] == 0, 1, 0)

    crash2["isex"] = np.where(crash2["isex"] == 1, 1, 0)

    # deal with missing data

    crash2 = crash2[(crash2.isbp.notnull()) & (crash2.isbp > 10)]  # 21
    crash2 = crash2[(crash2.irr.notnull()) & (crash2.irr > 5)]  # 149
    crash2 = crash2[
        (crash2["ninjurytime"] < 20) & (crash2["ninjurytime"] > 0)
    ]  # only 16 pts with ninjurytime > 20.

    crash2[treatment] = np.where(crash2[treatment] == "Active", 1, 0)
    crash2[outcome] = np.where(crash2["icause"].isna(), 1, 0)

    crash2 = crash2[
        crash2_continuous_vars + crash2_binary_vars + [treatment] + [outcome]
    ]

    # Load txa
    txa = pd.read_pickle("data/txa_cohort.pkl")
    all_year = pd.read_csv("data/all_year.csv", index_col=0, low_memory=False)

    txa["medatetime"] = pd.to_datetime(
        txa["medstartdate"].astype(str) + " " + txa["medstarttime"].astype(str),
        infer_datetime_format=True,
        errors="coerce",
    )
    all_year["injurydatetime_ref"] = pd.to_datetime(
        all_year["injurydate"].astype(str) + " " + all_year["injurytime"].astype(str),
        infer_datetime_format=True,
        errors="coerce",
    )

    txa = pd.merge(
        txa,
        all_year[["registryid", "iss", "injurydatetime_ref"]],
        on="registryid",
        how="left",
    )

    if "injurydatetime" in txa.columns:
        txa["injurydatetime"] = txa["injurydatetime"].fillna(txa["injurydatetime_ref"])
    else:
        txa["injurydatetime"] = txa["injurydatetime_ref"]

    txa = txa.drop(columns=["injurydatetime_ref"])

    txa["time_to_injury"] = (
        txa["medatetime"] - txa["injurydatetime"]
    ).dt.total_seconds() / 3600

    txa["time_to_injury"] = np.where(
        txa["time_to_injury"].isnull(),
        (txa["scenedatetime"] - txa["injurydatetime"]).dt.total_seconds() / 3600,
        txa["time_to_injury"],
    )

    # raw_data["iss"] = pd.to_numeric(raw_data["iss"], errors='coerce')

    filter_regex = [
        "proc",
        "ethnicity",
        "residencestate",
        "toxicologyresults",
        "registryid",
        "COV",
        "TT",
        "scenegcsmotor",
        "scenegcseye",
        "scenegcsverbal",
        "edgcsmotor",
        "edgcseye",
        "edgcsverbal",
        "sex_F",
        "traumatype_P",
        "traumatype_OTHER",
        "causecode",
    ]

    treatment = "treated"
    outcome = "outcome"

    for regex in filter_regex:
        txa = txa[txa.columns.drop(list(txa.filter(regex=regex)))]

    txa_continuous_vars = [
        "age",
        "scenefirstbloodpressure",
        "scenefirstrespirationrate",
        "scenefirstpulse",
        "scenegcs",
        "time_to_injury",
    ]

    txa_binary_vars = [
        "traumatype_B",
        "traumatype_P",
        "sex_M",
    ]
    txa["traumatype_P"] = np.where(txa["traumatype_B"] == 1, 0, 1)
    txa = txa[txa_continuous_vars + txa_binary_vars + [treatment] + [outcome]]
    txa = txa.rename(
        columns={
            "age": "iage",
            "scenefirstbloodpressure": "isbp",
            "scenefirstrespirationrate": "irr",
            "scenefirstpulse": "ihr",
            "scenegcs": "igcs",
            "time_to_injury": "ninjurytime",
            "traumatype_B": "iinjurytype_1",
            "traumatype_P": "iinjurytype_2",
            "sex_M": "isex",
            "treated": "treatment_code",
        }
    )

    txa = txa[txa.irr > 5]
    txa = txa[
        (txa.ninjurytime.notnull()) & (txa.ninjurytime > 0) & (txa.ninjurytime < 20)
    ]

    scaler = InvertableColumnTransformer(
        transformers=[
            # ('std', StandardScaler(),[
            # ]),
            (
                "minmax",
                MinMaxScaler(),
                ["iage", "isbp", "irr", "ihr", "igcs", "ninjurytime"],
            ),
            (
                "passthrough",
                "passthrough",
                ["iinjurytype_1", "iinjurytype_2", "isex", "treatment_code", "outcome"],
            ),
        ]
    )

    # Fit preprocessing on CRASH-2 only, then apply to both cohorts.
    crash2_scaled = scaler.fit_transform(crash2)
    txa_scaled = scaler.transform(txa)

    # Imputation is also fit on CRASH-2 reference distribution.
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(crash2_scaled)
    crash2 = imp.transform(crash2_scaled)
    txa = imp.transform(txa_scaled)

    # Propensity matching for TXA.
    X = txa[:, :-2]
    y = txa[:, -2]

    def _mean_abs_smd(group_a, group_b):
        """Return mean/max absolute standardized mean difference across covariates."""
        if len(group_a) == 0 or len(group_b) == 0:
            return np.nan, np.nan

        mean_diff = np.mean(group_a, axis=0) - np.mean(group_b, axis=0)
        var_pool = (np.var(group_a, axis=0, ddof=1) + np.var(group_b, axis=0, ddof=1)) / 2
        denom = np.sqrt(np.where(var_pool > 1e-12, var_pool, np.nan))
        smd = np.abs(mean_diff / denom)
        smd = np.nan_to_num(smd, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.mean(smd)), float(np.max(smd))

    # Propensity model using Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    propensity_score_logit = np.log(
        (model.predict_proba(X)[:, 1]) / (1 - model.predict_proba(X)[:, 1])
    )

    # Separate treated and control groups based on treatment column (-2 index)
    treated_indices = np.where(txa[:, -2] == 1)[0]
    control_indices = np.where(txa[:, -2] == 0)[0]
    treated_pre = txa[treated_indices, :-2]
    control_pre = txa[control_indices, :-2]
    txa = np.hstack([txa, propensity_score_logit[:, np.newaxis]])

    treated = txa[treated_indices, :]
    control = txa[control_indices, :]
    caliper = np.std(propensity_score_logit) * 0.25

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(control[:, -1].reshape(-1, 1))  # Propensity score is the last column

    # Find nearest neighbors for treated group and enforce caliper explicitly.
    distances, indices = nbrs.kneighbors(treated[:, -1].reshape(-1, 1))
    valid_mask = distances[:, 0] <= caliper

    treated = treated[valid_mask]
    matched_control = control[indices[valid_mask, 0]]

    if len(treated) == 0:
        raise ValueError("No treated samples have a control match within caliper.")

    treated_post = treated[:, :-3]
    control_post = matched_control[:, :-3]
    mean_smd_pre, max_smd_pre = _mean_abs_smd(treated_pre, control_pre)
    mean_smd_post, max_smd_post = _mean_abs_smd(treated_post, control_post)

    treated_total = len(valid_mask)
    treated_kept = int(valid_mask.sum())
    treated_dropped = treated_total - treated_kept
    unique_controls = int(np.unique(indices[valid_mask, 0]).shape[0])

    print("[obtain_txa_baselines] caliper matching diagnostics")
    print(f"  treated before: {treated_total}")
    print(f"  treated kept: {treated_kept}")
    print(f"  treated dropped: {treated_dropped} ({treated_dropped / max(treated_total, 1):.2%})")
    print(f"  matched controls (rows): {matched_control.shape[0]}")
    print(f"  unique matched controls: {unique_controls}")
    print(f"  mean abs SMD pre-match: {mean_smd_pre:.4f}")
    print(f"  mean abs SMD post-match: {mean_smd_post:.4f}")
    print(f"  max abs SMD pre-match: {max_smd_pre:.4f}")
    print(f"  max abs SMD post-match: {max_smd_post:.4f}")

    matched_txa = np.vstack([treated, matched_control])
    matched_txa = np.delete(matched_txa, -1, axis=1)

    if unnorm:
        crash2 = scaler.inverse_transform(crash2)
        matched_txa = scaler.inverse_transform(matched_txa)

    return (
        crash2[:, :-2],
        crash2[:, -2],
        crash2[:, -1],
        matched_txa[:, :-2],
        matched_txa[:, -2],
        matched_txa[:, -1],
    )


def obtain_unnorm_txa_baselines() -> np.ndarray:
    """Obtain unnormalzied txa baselines"""
    crash2 = pd.read_excel("data/crash_2.xlsx")

    outcome = "outcome"
    treatment = "treatment_code"

    crash2_continuous_vars = [
        "iage",
        "isbp",
        "irr",
        "ihr",
        "igcs",
    ]

    crash2_categorical_vars = ["iinjurytype"]

    crash2_binary_vars = ["isex"]

    crash2 = crash2.drop(
        crash2[(crash2[treatment] == "P") | (crash2[treatment] == "D")].index
    )

    crash2 = crash2[crash2.iinjurytype != 3]

    crash2["isex"] = np.where(crash2["isex"] == 2, 0, 1)

    # deal with missing data

    crash2["irr"] = np.where(crash2["irr"] == 0, np.nan, crash2["irr"])
    crash2["isbp"] = np.where(crash2["isbp"] == 999, np.nan, crash2["isbp"])

    crash2[treatment] = np.where(crash2[treatment] == "Active", 1, 0)
    crash2[outcome] = np.where(crash2["icause"].isna(), 1, 0)

    crash2 = crash2[
        crash2_continuous_vars
        + crash2_categorical_vars
        + crash2_binary_vars
        + [treatment]
        + [outcome]
    ]

    # Load txa
    txa = pd.read_pickle("data/txa_cohort.pkl")

    filter_regex = [
        "proc",
        "ethnicity",
        "residencestate",
        "toxicologyresults",
        "registryid",
        "COV",
        "TT",
        "scenegcsmotor",
        "scenegcseye",
        "scenegcsverbal",
        "edgcsmotor",
        "edgcseye",
        "edgcsverbal",
        "sex_F",
        "traumatype_P",
        "traumatype_OTHER",
        "causecode",
    ]

    treatment = "treated"
    outcome = "outcome"

    for regex in filter_regex:
        txa = txa[txa.columns.drop(list(txa.filter(regex=regex)))]

    txa_continuous_vars = [
        "age",
        "scenefirstbloodpressure",
        "scenefirstrespirationrate",
        "scenefirstpulse",
        "scenegcs",
    ]

    txa_binary_vars = [
        "traumatype_B",
        "sex_M",
    ]

    txa = txa[txa_continuous_vars + txa_binary_vars + [treatment] + [outcome]]

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(crash2)
    crash2 = imp.transform(crash2)

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(crash2)
    txa = imp.transform(txa)
    # Propensity matching for TXA.
    X = txa[:, :-2]
    y = txa[:, -2]

    # Propensity model using Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity_score = model.predict_proba(X)[:, 1]

    # Separate treated and control groups based on treatment column (-2 index)
    treated_indices = np.where(txa[:, -2] == 1)[0]
    control_indices = np.where(txa[:, -2] == 0)[0]
    txa = np.hstack([txa, propensity_score[:, np.newaxis]])

    treated = txa[treated_indices, :]
    control = txa[control_indices, :]

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(control[:, -1].reshape(-1, 1))  # Propensity score is the last column

    # Find nearest neighbors for treated group
    _, indices = nbrs.kneighbors(treated[:, -1].reshape(-1, 1))

    # Extract matched controls using the indices found
    matched_control = control[indices.flatten()]

    matched_txa = np.vstack([treated, matched_control])
    matched_txa = np.delete(matched_txa, -1, axis=1)

    return (
        crash2[:, :-2],
        crash2[:, -2],
        crash2[:, -1],
        matched_txa[:, :-2],
        matched_txa[:, -2],
        matched_txa[:, -1],
    )


def obtain_accord_baselines() -> np.ndarray:
    """
    Return normalized baseline of ACCORD dataset with SPRINT value range.

    Returns
    -------
        Tuple containing:
        - Normalized baselines for SPRINT
        - Treatment for SPRINT
        - Outcome for SPRINT
        - Normalized baselines for ACCORD
    """

    accord = pd.read_csv("data/accord/accord.csv")

    # Keep only features that have clear ACCORD↔SPRINT overlap.
    continuous_vars_accord = [
        "baseline_age",
        "sbp",
        "dbp",
        "gfr",
        "fpg",
        "hdl",
        "trig",
        "uacr",
        "bmi",
    ]

    binary_vars_accord = [
        "female",
        "raceclass",
        "x4smoke",
        "aspirin",
        "statin",
        "cvd_hx_baseline",
    ]

    accord["raceclass"] = np.where(accord["raceclass"] == "Black", 1, 0)
    accord["x4smoke"] = np.where(accord["x4smoke"] == 1, 1, 0)

    outcome = "censor_po"
    treatment = "treatment"

    accord = accord[
        continuous_vars_accord + binary_vars_accord + [treatment] + [outcome]
    ]
    accord["treatment"] = np.where(
        accord["treatment"].str.contains("Intensive BP", na=False), 1, 0
    )

    accord = accord.rename(
        columns={
            "baseline_age": "age",
            "gfr": "egfr",
            "fpg": "glur",
            "trig": "trr",
            "uacr": "umalcr",
            "raceclass": "race_black",
            "x4smoke": "smoke_3cat",
            "cvd_hx_baseline": "sub_cvd",
        }
    )

    outcome = pd.read_csv("data/sprint/outcomes.csv")
    baseline = pd.read_csv("data/sprint/baseline.csv")

    baseline.columns = [x.lower() for x in baseline.columns]
    outcome.columns = [x.lower() for x in outcome.columns]

    sprint = baseline.merge(outcome, on="maskid", how="inner")

    sprint["smoke_3cat"] = np.where(
        sprint["smoke_3cat"] == 4, np.nan, np.where(sprint["smoke_3cat"] == 3, 1, 0)
    )

    continuous_vars_sprint = [
        "age",
        "sbp",
        "dbp",
        "egfr",
        "glur",
        "hdl",
        "trr",
        "umalcr",
        "bmi",
    ]

    binary_vars_sprint = [
        "female",
        "race_black",
        "smoke_3cat",
        "aspirin",
        "statin",
        "sub_cvd",
    ]

    treatment = "intensive"
    outcome_col = "event_primary"

    sprint = sprint[
        continuous_vars_sprint + binary_vars_sprint + [treatment] + [outcome_col]
    ]
    sprint[outcome_col] = np.where(sprint[outcome_col] == 1, 0, 1)

    scaler = MinMaxScaler()

    scaler.fit(
        np.concatenate(
            (
                sprint[continuous_vars_sprint].values,
                accord[continuous_vars_sprint].values,
            ),
            axis=0,
        )
    )

    sprint[continuous_vars_sprint] = scaler.transform(
        sprint[continuous_vars_sprint].values
    )
    accord[continuous_vars_sprint] = scaler.transform(
        accord[continuous_vars_sprint].values
    )

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(sprint)
    sprint = imp.transform(sprint)

    imp.fit(accord)
    accord = imp.transform(accord)

    return (
        sprint[:, :-2],
        sprint[:, -2],
        sprint[:, -1],
        accord[:, :-2],
        accord[:, -2],
        accord[:, -1],
    )


def obtain_unnorm_accord_baselines() -> np.ndarray:
    """
    Return normalized baseline of ACCORD dataset with SPRINT value range.

    Returns
    -------
        Tuple containing:
        - Normalized baselines for SPRINT
        - Treatment for SPRINT
        - Outcome for SPRINT
        - Normalized baselines for ACCORD
    """

    accord = pd.read_csv("data/accord/accord.csv")

    # Keep only features that have clear ACCORD↔SPRINT overlap.
    continuous_vars_accord = [
        "baseline_age",
        "sbp",
        "dbp",
        "gfr",
        "fpg",
        "hdl",
        "trig",
        "uacr",
        "bmi",
    ]

    binary_vars_accord = [
        "female",
        "raceclass",
        "x4smoke",
        "aspirin",
        "statin",
        "cvd_hx_baseline",
    ]

    accord["raceclass"] = np.where(accord["raceclass"] == "Black", 1, 0)
    accord["x4smoke"] = np.where(accord["x4smoke"] == 1, 1, 0)

    outcome = "censor_po"
    treatment = "treatment"

    accord = accord[
        continuous_vars_accord + binary_vars_accord + [treatment] + [outcome]
    ]
    accord["treatment"] = np.where(
        accord["treatment"].str.contains("Intensive BP", na=False), 1, 0
    )

    accord = accord.rename(
        columns={
            "baseline_age": "age",
            "gfr": "egfr",
            "fpg": "glur",
            "trig": "trr",
            "uacr": "umalcr",
            "raceclass": "race_black",
            "x4smoke": "smoke_3cat",
            "cvd_hx_baseline": "sub_cvd",
        }
    )

    outcome = pd.read_csv("data/sprint/outcomes.csv")
    baseline = pd.read_csv("data/sprint/baseline.csv")

    baseline.columns = [x.lower() for x in baseline.columns]
    outcome.columns = [x.lower() for x in outcome.columns]

    sprint = baseline.merge(outcome, on="maskid", how="inner")

    sprint["smoke_3cat"] = np.where(
        sprint["smoke_3cat"] == 4, np.nan, np.where(sprint["smoke_3cat"] == 3, 1, 0)
    )

    continuous_vars_sprint = [
        "age",
        "sbp",
        "dbp",
        "egfr",
        "glur",
        "hdl",
        "trr",
        "umalcr",
        "bmi",
    ]

    binary_vars_sprint = [
        "female",
        "race_black",
        "smoke_3cat",
        "aspirin",
        "statin",
        "sub_cvd",
    ]

    treatment = "intensive"
    outcome_col = "event_primary"

    sprint = sprint[
        continuous_vars_sprint + binary_vars_sprint + [treatment] + [outcome_col]
    ]
    sprint[outcome_col] = np.where(sprint[outcome_col] == 1, 0, 1)

    # scaler = MinMaxScaler()

    # scaler.fit(
    #     np.concatenate(
    #         (
    #           sprint[continuous_vars_sprint].values,
    #           accord[continuous_vars_accord].values
    #         ), axis=0
    #     )
    # )

    # sprint[continuous_vars_sprint] = scaler.transform(
    # sprint[continuous_vars_sprint].values)
    # accord[continuous_vars_accord] = scaler.transform(
    # accord[continuous_vars_accord].values)

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(sprint)
    sprint = imp.transform(sprint)

    imp.fit(accord)
    accord = imp.transform(accord)

    return (
        sprint[:, :-2],
        sprint[:, -2],
        sprint[:, -1],
        accord[:, :-2],
        accord[:, -2],
        accord[:, -1],
    )
