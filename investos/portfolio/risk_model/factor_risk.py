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
            self._drop_excluded_assets()
            self._apply_risk_model_adjustments()
            print("\nDone generating point-in-time structural risk models.")

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

    def portfolio_variance_estimate(self, t, w_plus):
        if self._dynamic_vra:
            factor_covar = util.values_in_time(
                self.adj_factor_covariance, t, lookback_for_closest=True
            )
            idiosync_var = util.values_in_time(
                self.adj_idiosyncratic_variance, t, lookback_for_closest=True
            )
        else:
            factor_covar = util.values_in_time(
                self.factor_covariance, t, lookback_for_closest=True
            )
            idiosync_var = util.values_in_time(
                self.idiosyncratic_variance, t, lookback_for_closest=True
            )

        factor_load = util.values_in_time(
            self.factor_loadings, t, lookback_for_closest=True
        )

        w = w_plus.values  # n x 1
        B = factor_load.values  # n x k
        F = factor_covar.values  # k x k
        D_diag = idiosync_var.values  # n x 1

        # Factor risk: (B^T w)^T F (B^T w)
        factor_exposure = B @ w  # k x 1
        factor_risk = factor_exposure.T @ F @ factor_exposure  # scalar

        # Idiosyncratic risk: w^T D w
        idiosyncratic_risk = np.sum((w**2) * D_diag)  # scalar

        # Total portfolio variance
        portfolio_variance = factor_risk + idiosyncratic_risk
        portfolio_variance = min(portfolio_variance, self.max_daily_var_est)

        # Convert to desired variance time period
        portfolio_variance = portfolio_variance * self._periods_in_estimate

        return portfolio_variance

    def portfolio_volatility_estimate(self, t, w_plus):
        return self.portfolio_variance_estimate(t, w_plus) ** 0.5

    def _make_dynamic_volatility_regime_adjustment(self, weights_for_vra_adj):
        """
        Volatility Regime Adjustment (VRA)
        """

        print("Making dynamic volatility regime adjustment.")

        # --------------------------------------------------------------------
        # Helper: EWMA with a half-life measured in periods (trading days)
        # --------------------------------------------------------------------
        def ewm_half_life(series: pd.Series, half_life: int) -> pd.Series:
            """Exponentially weighted moving average with an intuitive half-life."""
            alpha = 1.0 - np.exp(np.log(0.5) / half_life)
            return series.ewm(alpha=alpha, adjust=False).mean()

        # --------------------------------------------------------------------
        # 1.  Reconstruct *realised* factor & specific returns from X_t
        # --------------------------------------------------------------------
        asset_returns = self.actual_returns
        dates = asset_returns.index

        cols = [f"{col}_returns" for col in list(self.factor_covariance.columns.values)]
        factor_rts = self.factor_returns[["datetime", *cols]].set_index("datetime")
        specific_rts = self.idiosyncratic_returns.pivot(
            columns="id", index="datetime", values="idio_return"
        )[self.idiosyncratic_variance.columns]

        # --------------------------------------------------------------------
        # 2.  Pull forecast *volatilities* (σ) from the inputs
        # --------------------------------------------------------------------
        factor_sigma_fc = pd.DataFrame(  # (T × K)
            {
                d: np.sqrt(np.diag(self.factor_covariance.loc[d]))
                for d in self.factor_covariance.index.get_level_values(0)
            }
        ).T

        specific_sigma_fc = self.idiosyncratic_variance.apply(np.sqrt)  # (T × N)

        # --------------------------------------------------------------------
        # 3.  Standardise realised returns by forecasts  → z-scores
        # --------------------------------------------------------------------
        # Backwards merge_asof forecasts (fc) to actuals index
        factor_sigma_fc = pd.merge_asof(
            pd.DataFrame(index=factor_rts.index).reset_index(),
            factor_sigma_fc.reset_index().rename(columns={"index": "datetime"}),
            on="datetime",  # or on='date' if you renamed it
            direction="backward",
        ).set_index("datetime")

        specific_sigma_fc = pd.merge_asof(
            pd.DataFrame(index=specific_rts.index).reset_index(),
            specific_sigma_fc.reset_index().rename(columns={"index": "datetime"}),
            on="datetime",
            direction="backward",
        ).set_index("datetime")

        factor_rts.columns = factor_sigma_fc.columns

        z_factor = factor_rts / factor_sigma_fc
        z_specific = specific_rts / specific_sigma_fc

        # Ffill where holiday
        holiday_dates = specific_rts[
            (specific_rts.isna() | (specific_rts == 0)).all(axis=1)
        ].index

        z_factor.loc[holiday_dates] = np.nan
        z_factor = z_factor.ffill(limit=1)
        z_factor = z_factor.dropna(how="all")

        z_specific.loc[holiday_dates] = np.nan
        z_specific = z_specific.ffill(limit=1)
        z_specific = z_specific.dropna(how="all")

        # --------------------------------------------------------------------
        # 4.  Cross-sectional bias statistics, B_t
        #     – equal-weight across factors
        #     – cap-weight across stocks
        # --------------------------------------------------------------------
        B_factor_daily = (z_factor**2).mean(axis=1)
        weights = weights_for_vra_adj  # w_{n,t}
        weights = weights.reindex(z_specific.index, method="bfill")
        B_spec_daily = (z_specific**2 * weights).sum(axis=1)

        # # --------------------------------------------------------------------
        # # 5.  Smooth B_t through time  → λ²_t   and λ_t
        # # --------------------------------------------------------------------
        λ2_factor = ewm_half_life(B_factor_daily, self._dynamic_vra_config["half_life"])
        λ2_spec = ewm_half_life(B_spec_daily, self._dynamic_vra_config["half_life"])

        λ_factor, λ_spec = np.sqrt(λ2_factor), np.sqrt(λ2_spec)  # Series (T,)

        # # Optional: clip multipliers for robustness
        self._λ_factor = λ_factor.clip(upper=self._dynamic_vra_config["max_mult"])
        self._λ_spec = λ_spec.clip(upper=self._dynamic_vra_config["max_mult"])

        # --------------------------------------------------------------------
        # 6.  Produce *next period's (tomorrow's)* adjusted forecasts
        # --------------------------------------------------------------------
        # a) Factor covariance matrices – scale entire Σ_F by λ_factor² (correlations unchanged)
        dates = dates[dates >= self.start_date]

        factor_covar_adj_dfs = []
        idio_var_adj_dfs = []
        for t_idx in range(len(dates) - 1):
            today, tomorrow = dates[t_idx], dates[t_idx + 1]
            tmp = util.values_in_time(
                self.factor_covariance, tomorrow, lookback_for_closest=True
            ) * (self._λ_factor.loc[today] ** 2)
            tmp.index = pd.MultiIndex.from_product(
                [[tomorrow], tmp.index], names=["datetime", "index"]
            )
            factor_covar_adj_dfs.append(tmp)

            # b) Idiosyncratic variances – scale by λ_spec²
            tmp = util.values_in_time(
                self.idiosyncratic_variance, tomorrow, lookback_for_closest=True
            ) * (self._λ_spec.loc[today] ** 2)
            idio_var_adj_dfs.append([tomorrow, tmp])

        self.adj_factor_covariance = pd.concat(factor_covar_adj_dfs)
        self.adj_idiosyncratic_variance = pd.concat(
            {date: series for date, series in idio_var_adj_dfs}, axis=1
        ).T

    def _get_fos_risk_factor_data(self):
        print("Getting risk factor data from ForecastOS:")

        # Start with returns
        print(f"- Getting {self._fos_return_name}")
        self.actual_returns = (
            fos.Feature.get(self._fos_return_uuid)
            .get_df()
            .rename(columns={"value": self._fos_return_name})
        )
        self.actual_returns = risk_util.drop_na_and_inf(self.actual_returns)
        self.factor_loadings = self.actual_returns.copy()
        self.actual_returns = (
            self.actual_returns.pivot(columns="id", index="datetime")
            .sort_index(axis=1)
            .sort_index(axis=0)
        )
        self.actual_returns = self.actual_returns[
            (
                self.actual_returns.index
                >= (
                    self.start_date
                    - self._risk_model_window_td
                    - self._dynamic_vra_config["td_padding"]
                )
            )
            & (self.actual_returns.index <= self.end_date)
        ]
        self.actual_returns.columns = self.actual_returns.columns.droplevel(0)

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
                >= (
                    self.start_date
                    - self._risk_model_window_td
                    - self._dynamic_vra_config["td_padding"]
                )
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
            (datetimes >= self.start_date - self._dynamic_vra_config["td_padding"])
            & (datetimes <= self.end_date)
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
        if self._dynamic_vra:
            weights = self._dynamic_vra_config["weights"]
            if not weights:
                df_market_cap = fos.get_feature_df(
                    "dfa7e6a3-671d-41b2-89e3-10b7bdcf7af9"
                )
                df_market_cap = df_market_cap.dropna(subset=["value"])

                df_market_cap["datetime"] = pd.to_datetime(df_market_cap["datetime"])

                # Rank within each datetime by descending value
                df_market_cap["rank"] = df_market_cap.groupby("datetime")["value"].rank(
                    ascending=False, method="dense"
                )

                # Optional: Convert to int if you prefer integer ranks
                df_market_cap["rank"] = df_market_cap["rank"].astype(int)

                # Keep selected dates only
                df_market_cap = df_market_cap[df_market_cap.datetime >= self.start_date]
                df_market_cap = df_market_cap[df_market_cap.datetime <= self.end_date]

                # Remove errant data for L14CLL-R (Faraday Future Intelligent Electric)
                df_market_cap = df_market_cap[df_market_cap.id != "L14CLL-R"]

                df_market_cap = df_market_cap[df_market_cap["rank"] <= 100]
                df_market_cap["pct_value"] = df_market_cap[
                    "value"
                ] / df_market_cap.groupby("datetime")["value"].transform("sum")

                weights = (
                    df_market_cap[["id", "datetime", "pct_value"]]
                    .pivot(index="datetime", columns="id", values="pct_value")
                    .fillna(0)
                    .sort_index(axis=1)
                )

            self._make_dynamic_volatility_regime_adjustment(weights)

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
        self.actual_returns = self._remove_excl_columns(self.actual_returns)

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
            "risk_model_window_td", timedelta(days=325)
        )
        self._recalc_td = kwargs.get("recalc_td", timedelta(days=21))
        self._periods_in_estimate = kwargs.get(
            "periods_in_estimate", 252
        )  # For making yearly (~252), monthly (~21), or daily (1)
        self._penalize_risk = kwargs.get("penalize_risk", False)
        self._max_std_dev = kwargs.get("max_std_dev", None)
        self._dynamic_vra = kwargs.get("dynamic_vra", True)
        self._dynamic_vra_config = {
            "half_life": 7,
            "max_mult": 3.0,
            "td_padding": timedelta(days=365),
            "weights": False,
        }
        self._dynamic_vra_config.update(kwargs.get("dynamic_vra_config", {}))
        self.max_daily_var_est = kwargs.get("max_daily_var_est", 0.00063)
        if self._max_std_dev:
            print(
                "\nMake sure you are using a solver that can handle quadratic constraints (since you set max standard deviation in your risk model, which creates a quadratic constraint), like cvx.CLARABEL.\nNote that cvx.OSQP doesn't support convex constraints as of Aug 2024."
            )
