import re
import warnings
from datetime import datetime, timedelta

import cvxpy as cvx
import forecastos as fos
import numpy as np
import pandas as pd
import statsmodels.api as sm

import investos.portfolio.risk_model.risk_util as risk_util
import investos.util as util
from investos.portfolio.risk_model import BaseRisk


class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self,
        factor_covariance=None,
        factor_loadings=None,
        idiosyncratic_variance=None,
        fos_risk_factor_uuids_dict: dict = risk_util.fos_risk_factor_uuids_dict,
        fos_risk_factor_adj_dict: dict = risk_util.fos_risk_factor_adj_dict,
        fos_return_uuid: str = risk_util.fos_return_uuid,
        fos_return_name: str = risk_util.fos_return_name,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if factor_covariance and factor_loadings and idiosyncratic_variance:
            print("Using provided point-in-time structural risk models.\n")
            self.factor_covariance = factor_covariance
            self.factor_loadings = factor_loadings
            self.idiosyncratic_variance = idiosyncratic_variance
        else:
            self._fos_return_uuid = fos_return_uuid
            self._fos_return_name = fos_return_name
            self._fos_risk_factor_uuids_dict = fos_risk_factor_uuids_dict
            self._fos_risk_factor_adj_dict = fos_risk_factor_adj_dict
            self._set_optional_args(**kwargs)

            print("Generating point-in-time structural risk models.\n")
            self._get_fos_risk_factor_data()
            self._wins_fill_std_factor_data()
            if self._add_global_constant_factor:
                self.factor_loadings["global_const"] = 1
            self._calculate_factor_returns()
            self._calculate_idiosyncratic_returns()
            self._generate_risk_models()
            self._apply_risk_model_adjustments()
            print("\nDone generating point-in-time structural risk models.")

        self._drop_excluded_assets()

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated costs for backtest performance.
        """
        factor_covar = util.values_in_time(
            self.factor_covariance, t, lookback_for_closest=True
        )
        factor_load = util.values_in_time(
            self.factor_loadings, t, lookback_for_closest=True
        )
        idiosync_var = util.values_in_time(
            self.idiosyncratic_variance, t, lookback_for_closest=True
        )

        risk_from_factors = factor_load.T @ factor_covar @ factor_load
        sigma = risk_from_factors + np.diag(idiosync_var)
        self.expression = cvx.quad_form(w_plus, sigma)

        if self._penalize_risk:
            risk_penalty = self.expression
        else:
            risk_penalty = cvx.sum(0)

        if self._max_std_dev:
            constr_li = [self.expression <= (self._max_std_dev**2)]
        else:
            constr_li = []

        return risk_penalty, constr_li

    def _get_fos_risk_factor_data(self):
        print("Getting risk factor data from ForecastOS:")

        # Start with returns
        print(f"- Getting {self._fos_return_name}")
        self.factor_loadings = (
            fos.Feature.get(self._fos_return_uuid)
            .get_df()
            .rename(columns={"value": self._fos_return_name})
        )
        self.factor_loadings = risk_util.drop_na_and_inf(self.factor_loadings)

        # Join risk factors
        for factor, uuid in self._fos_risk_factor_uuids_dict.items():
            print(f"- Getting {factor}")
            temp_df = risk_util.drop_na_and_inf(
                fos.Feature.get(uuid).get_df().rename(columns={"value": factor})
            )

            self.factor_loadings = pd.merge_asof(
                self.factor_loadings.sort_values("datetime"),
                temp_df.sort_values("datetime"),
                on="datetime",
                by="id",
                direction="backward",
                tolerance=pd.Timedelta("150 days"),
            )

        self.factor_loadings = self.factor_loadings[
            (
                self.factor_loadings.datetime
                >= (self.start_date - self._risk_model_window_td)
            )
            & (self.factor_loadings.datetime <= self.end_date)
        ]

        print("\nDone getting factor data.")

    def _wins_fill_std_factor_data(self):
        print(
            "\nWinsorizing, mean-filling, and standardizing risk factor data (grouped by datetime)..."
        )
        warnings.filterwarnings("ignore", category=UserWarning)
        self.factor_loadings = self.factor_loadings.groupby("datetime").apply(
            risk_util.wins_std_mean_fill,
            [self._fos_return_name],  # Don't standardize
            self._fos_risk_factor_adj_dict,
            include_groups=False,
        )
        print("Done winsorizing, mean-filling, and standardizing risk factor data.")

    def _calculate_factor_returns(self):
        print("\nCalculating factor returns for each period:\n")
        self.factor_returns = {}
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )

        for dt in sorted(self.factor_loadings.index.get_level_values(0).unique()):
            df_current = self.factor_loadings.loc[dt]

            factor_cols = self._get_factor_cols()

            return_y, factors_X = (
                df_current[[self._fos_return_name]],
                df_current[[*factor_cols]],
            )

            model = sm.OLS(return_y, factors_X)
            results = model.fit()

            # store coefficients, intercept, feature_names, r2, and t-values into self._factor_returns based on the date
            self.factor_returns[dt] = {
                "returns": results.params,
                "t_values": results.tvalues,
                "p_values": results.pvalues,
                "r2": results.rsquared_adj,
                "f_value": results.fvalue,
                "f_pvalue": results.f_pvalue,
            }
            print(".", end="")

        self._create_factor_returns_df()

        print("\n\nDone calculating factor returns for each period.")

    def _calculate_idiosyncratic_returns(self):
        factor_return_cols = list(
            [col for col in self.factor_returns.columns if col.endswith("_returns")]
        )

        # Merge factor loadings and returns
        self.idiosyncratic_returns = pd.merge(
            self.factor_loadings.reset_index(),
            self.factor_returns[["datetime"] + factor_return_cols],
            on="datetime",
            how="left",
        )

        # Calculate factor returns
        self.idiosyncratic_returns["cum_factor_return"] = 0
        for factor_return_col in factor_return_cols:
            factor_loading_col = re.sub(r"_returns$", "", factor_return_col)
            self.idiosyncratic_returns["cum_factor_return"] = (
                self.idiosyncratic_returns["cum_factor_return"]
                + self.idiosyncratic_returns[factor_return_col]
                * self.idiosyncratic_returns[factor_loading_col]
            )

        # Calculate idiosyncratic returns using actual vs factor returns
        self.idiosyncratic_returns["idio_return"] = (
            self.idiosyncratic_returns[self._fos_return_name]
            - self.idiosyncratic_returns["cum_factor_return"]
        )

        # Keep only idio returns and cum factor returns
        self.idiosyncratic_returns = self.idiosyncratic_returns[
            [
                "datetime",
                "id",
                "idio_return",
                "cum_factor_return",
                self._fos_return_name,
            ]
        ]

    def _generate_risk_models(self):
        self._create_empty_risk_models()

        datetimes = self.factor_loadings.index.get_level_values(0).unique()
        datetimes = datetimes[
            (datetimes >= self.start_date) & (datetimes <= self.end_date)
        ]

        factor_loadings_tmp_df = self.factor_loadings.reset_index().drop(
            columns=["level_1"]
        )
        self._reformat_factor_loadings_df()
        last_risk_model_dt = None

        print("\nGenerating idiosyncratic variance and factor covariance DFs for:\n")
        for dt in datetimes:
            if not last_risk_model_dt or (dt - last_risk_model_dt) >= self._recalc_td:
                last_risk_model_dt = dt
                print(dt.date(), end=" ")

                window_factor_loadings_df = factor_loadings_tmp_df[
                    (
                        factor_loadings_tmp_df.datetime
                        >= (dt - self._risk_model_window_td)
                    )
                    & (factor_loadings_tmp_df.datetime < dt)
                ].dropna()

                window_factor_returns_df = self.factor_returns[
                    (self.factor_returns.datetime >= (dt - self._risk_model_window_td))
                    & (self.factor_returns.datetime < dt)
                ].dropna()

                self._calculate_idiosyncratic_variance(window_factor_loadings_df, dt)
                self._calculate_factor_covariance(window_factor_returns_df, dt)

        self._zero_fill_var_df()  # For companies no longer trading on datetimes

        print("\n\nDone generating idiosyncratic variance and factor covariance DFs.")

    def _calculate_idiosyncratic_variance(self, window_df, dt):
        window_df = window_df[["id", "datetime", self._fos_return_name]]
        var_s = window_df.groupby("id")[self._fos_return_name].var().sort_index()
        var_df = pd.DataFrame([var_s.values], columns=var_s.index, index=[dt]).dropna(
            axis=1, how="all"
        )

        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.idiosyncratic_variance = pd.concat([self.idiosyncratic_variance, var_df])

    def _calculate_factor_covariance(self, window_factor_returns_df, dt):
        # Prepare factor returns df
        factor_cols = self._get_return_cols_from_df(window_factor_returns_df)
        window_factor_returns_df = window_factor_returns_df.dropna()[
            ["datetime", *factor_cols]
        ]
        window_factor_returns_df.columns = window_factor_returns_df.columns.str.replace(
            "_returns", ""
        )

        # Calculate factor covariance
        cov_df = window_factor_returns_df.drop(columns=["datetime"]).cov()

        # Check for PSD
        eigenvalues = np.linalg.eigvals(cov_df)
        is_positive_semi_definite = np.all(eigenvalues >= 0)
        if not is_positive_semi_definite:
            raise ValueError(
                f"The factor covariance matrix is not positive semi-definite for {dt}."
            )

        # Clean factor covariance
        cov_df = cov_df.reset_index()
        cov_df["datetime"] = dt
        cov_df = (
            cov_df.set_index(["datetime", "index"])
            .sort_index(axis=1)
            .sort_index(axis=0)
        )

        # Set factor covariance
        if hasattr(self, "factor_covariance"):
            self.factor_covariance = pd.concat([self.factor_covariance, cov_df])
        else:
            self.factor_covariance = cov_df

    def _reformat_factor_loadings_df(self):
        factor_cols = self._get_factor_cols()
        self.factor_loadings = self.factor_loadings.reset_index().drop(
            columns=["level_1"]
        )[["id", "datetime", *factor_cols]]

        self.factor_loadings = self.factor_loadings.melt(
            id_vars=["id", "datetime"], var_name="factor"
        )
        self.factor_loadings = (
            (
                self.factor_loadings.pivot(
                    columns="id", index=["datetime", "factor"]
                ).fillna(0)
            )
            .sort_index(axis=1)
            .sort_index(axis=0)
        )

        self.factor_loadings.columns = self.factor_loadings.columns.droplevel(0)

    def _apply_risk_model_adjustments(self):
        pass

    def _create_empty_risk_models(self):
        self.idiosyncratic_variance = pd.DataFrame(
            columns=self.factor_loadings.id.unique()
        ).sort_index(axis=1)
        self.idiosyncratic_variance.index.name = "datetime"

    def _zero_fill_var_df(self):
        self.idiosyncratic_variance = self.idiosyncratic_variance.fillna(0)

    def _drop_excluded_assets(self):
        self.factor_loadings = self._remove_excl_columns(self.factor_loadings)
        self.idiosyncratic_variance = self._remove_excl_columns(
            self.idiosyncratic_variance
        )

    def _create_factor_returns_df(self):
        df_list = []

        for date, values in self.factor_returns.items():
            row = {
                "datetime": date,
                "r2": values["r2"],
                "f_value": values["f_value"],
                "f_pvalue": values["f_pvalue"],
            }
            for idx_key in values["returns"].index:
                row[f"{idx_key}_returns"] = values["returns"][idx_key]
                row[f"{idx_key}_t_value"] = values["t_values"][idx_key]
                row[f"{idx_key}_p_value"] = values["p_values"][idx_key]

            df_list.append(row)

        self.factor_returns = pd.DataFrame(df_list)

    def _get_return_cols_from_df(self, df):
        return list([col for col in df.columns if col.endswith("_returns")])

    def _remove_excl_columns(self, pd_obj):
        return util.remove_excluded_columns_pd(
            pd_obj,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

    def _get_factor_cols(self):
        factor_cols = list(self._fos_risk_factor_uuids_dict.keys())
        if self._add_global_constant_factor:
            factor_cols += ["global_const"]

        return factor_cols

    def _set_optional_args(self, **kwargs):
        self.adjustments = kwargs.get("adjustments", {})
        self.start_date = kwargs.get(
            "start_date", datetime.now() - timedelta(days=6 * 365)
        )  # 6 years ago, by default
        self.end_date = kwargs.get(
            "end_date", datetime.now() + timedelta(days=1)
        )  # Clip after tomorrow, by default

        self._add_global_constant_factor = kwargs.get(
            "add_global_constant_factor", True
        )
        self._risk_model_window_td = kwargs.get(
            "risk_model_window_td", timedelta(days=91)
        )
        self._recalc_td = kwargs.get("recalc_td", timedelta(days=30))
        self._ppy_for_annualizing_var = kwargs.get("ppy_for_annualizing_var", 252)
        self._penalize_risk = kwargs.get("penalize_risk", True)
        self._max_std_dev = kwargs.get("max_std_dev", None)
        if self._max_std_dev:
            print(
                "\nMake sure you are using a solver that can handle quadratic constraints (since you set max standard deviation in your risk model, which creates a quadratic constraint), like cvx.CLARABEL.\nNote that cvx.OSQP doesn't support convex constraints as of Aug 2024."
            )
