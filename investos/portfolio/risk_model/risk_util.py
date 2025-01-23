import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

fos_return_uuid = "ea4d2557-7f8f-476b-b4d3-55917a941bb5"
fos_return_name = "return_1d"

fos_risk_factor_uuids_dict = {
    # Beta
    "beta_24m": "79005629-e5a9-40ff-b677-b1278c6fa366",
    # Momentum
    "return_252d": "ed02d053-a0e1-4447-aff2-ad399f770f14",
    # Size
    "market_cap_open_dil": "dfa7e6a3-671d-41b2-89e3-10b7bdcf7af9",
    # Quality (margin)
    "net_income_div_sales_ltm": "7f1e058f-46b6-406f-81a1-d8a5b81371a2",
    # Growth
    "sales_ltm_growth_over_sales_ltm_lag_1a": "e6035b0a-65b9-409d-a02e-b3d47ca422e2",
    # Leverage
    "debt_total_prev_1q_to_ebit_ltm": "53a422bf-1dab-4d1e-a9a7-2478a226435b",
    # Value
    "market_cap_open_dil_to_operating_excl_wc_cf_ltm": "5f050fce-5099-4ce9-a737-5d8f536c5826",  # ltm market_cap_open_dil to operating_excl_wc_cf multiple. ID is FactSet Sym ID.
}
fos_risk_factor_adj_dict = {
    "normalization": {
        "market_cap_open_dil": [np.log],
    },
    "winsorization": {
        "return_1d": [0.01, 0.01],
        "net_income_div_sales_ltm": [0.2, 0.2],
        "sales_ltm_growth_over_sales_ltm_lag_1a": [0.2, 0.2],
    },
}
winsorization_default = [0.10, 0.10]


def drop_na_and_inf(df: pd.DataFrame):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def wins_std_mean_fill(group, dont_std_cols, adj_dict, drop_cols=["id"]):
    cols = group.columns.drop(drop_cols)
    imputer = SimpleImputer(strategy="mean")

    for col in cols:
        # Winsorizing numeric columns within the group
        wins_limits = adj_dict.get("winsorization", {}).get(col, winsorization_default)
        group[col] = winsorize(group[col], limits=wins_limits)
        group[col] = imputer.fit_transform(group[[col]])

    for col in cols:
        for func in adj_dict.get("normalization", {}).get(col, []):
            group[col] = func(group[col])

    cols = [col for col in cols if col not in dont_std_cols]

    # Standardizing numeric columns within the group
    scaler = StandardScaler()
    group[cols] = scaler.fit_transform(group[cols])

    return group
