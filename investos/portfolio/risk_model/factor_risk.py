import cvxpy as cvx
import forecastos as fos
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler

import investos.util as util
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.risk_model.utils.risk_adjustments import (
    FactorCovarianceProcessor,
    IdiosyncraticVarianceProcessor,
)
from investos.portfolio.risk_model.utils.risk_utils import clean_na_and_inf

fos_return_uuid_dict = {
    "return_1d": "ea4d2557-7f8f-476b-b4d3-55917a941bb5",
}
fos_risk_factor_uuids_dict = {
    # Size
    "market_cap_open_dil": "dfa7e6a3-671d-41b2-89e3-10b7bdcf7af9",
    # Leverage
    "debt_total_prev_1q_to_ebit_ltm": "53a422bf-1dab-4d1e-a9a7-2478a226435b",
    # Quality
    "net_income_div_sales_ltm": "7f1e058f-46b6-406f-81a1-d8a5b81371a2",
    # Value
    "ev_to_ebit_ltm": "f65f7b6a-5433-4543-8f15-5f50e43dd9f9",
    # Growth
    "sales_ltm_growth_over_sales_ltm_lag_1a": "e6035b0a-65b9-409d-a02e-b3d47ca422e2",
}


class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self,
        start_date,
        end_date,
        risk_model_window: int,
        recalc_freq: int,
        fos_risk_factor_uuids_dict: dict = fos_risk_factor_uuids_dict,
        fos_return_uuid_dict: dict = fos_return_uuid_dict,
        adjustments: dict = {"FactorCovEstArgs": {"window": 21, "half_life": 360}},
        factor_covariance=None,
        factor_loadings=None,
        idiosyncratic_variance=None,
        add_global_constant_factor=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._risk_model_window = risk_model_window
        self._recalc_freq = recalc_freq

        self._fos_risk_factor_uuids_dict = fos_risk_factor_uuids_dict
        self._fos_return_uuid_dict = fos_return_uuid_dict

        self._end_date = end_date
        self._start_date = start_date
        self._adjustments = adjustments

        self._add_global_constant_factor = add_global_constant_factor
        if self._fos_risk_factor_uuids_dict:
            print("Generating Structural Risk Model...")
            self._pull_fos_risk_factor_dfs()
            self._generate_risk_models()
            self._apply_risk_model_adjustments()
            print("Done generating Structural Risk Model")
        else:
            self.factor_covariance = factor_covariance
            self.factor_loadings = factor_loadings
            self.idiosyncratic_variance = idiosyncratic_variance

    def _pull_fos_risk_factor_dfs(self):
        print("Getting factor data from ForecastOS FeatureHub...")

        if not fos.api_key:
            raise ValueError(
                "Feature hub connection failed. Please check the API key is set."
            )

        self._fos_return_uuid = list(self._fos_return_uuid_dict.keys())[0]
        self._df_loadings = (
            fos.Feature.get(self._fos_return_uuid_dict[self._fos_return_uuid])
            .get_df()
            .rename(columns={"value": self._fos_return_uuid})
        )

        self._fos_risk_factor_uuids = list(self._fos_risk_factor_uuids_dict.keys())
        for factor in self._fos_risk_factor_uuids_dict.keys():
            temp_df = (
                fos.Feature.get(self._fos_risk_factor_uuids_dict[factor])
                .get_df()
                .rename(columns={"value": factor})
            )

            self._df_loadings = pd.merge_asof(
                self._df_loadings.sort_values("datetime"),
                temp_df.sort_values("datetime"),
                on="datetime",
                by="id",
                direction="forward",
            )
        self._df_loadings = clean_na_and_inf(self._df_loadings)

        self._df_loadings = self._df_loadings[
            (self._df_loadings.datetime >= self._end_date)
            & (self._df_loadings.datetime <= self._start_date)
        ]

        print("Done getting factor data")
        return

    def _generate_risk_models(self, datetime_col="datetime") -> None:
        """
        fit a cross sectional factor model and generate factor return and idio returns
        """

        """
        fit the cross sectional regression
        """
        print("Fitting Cross Sectional Regression...")
        scaler = StandardScaler()
        self._factor_returns = {}
        if self._add_global_constant_factor:
            self._df_loadings.loc[:, "global_const"] = 1
            self._fos_risk_factor_uuids.append("global_const")

        for date in sorted(self._df_loadings[datetime_col].unique()):
            # isolating the data for the current date
            df_current = self._df_loadings[self._df_loadings[datetime_col] == date]

            # winsorzie and standarize risk factors
            for col in self._fos_risk_factor_uuids:
                df_current.loc[:, col] = mstats.winsorize(
                    df_current.loc[:, col], limits=(0.05, 0.95)
                )
                df_current.loc[:, col] = scaler.fit_transform(
                    df_current.loc[:, col].values.reshape(-1, 1)
                ).flatten()
            # winsorzie return
            df_current.loc[:, self._fos_return_uuid] = mstats.winsorize(
                df_current.loc[:, self._fos_return_uuid], limits=(0.05, 0.95)
            )

            return_y, factors_X = (
                df_current.loc[:, self._fos_return_uuid],
                df_current.loc[:, self._fos_risk_factor_uuids],
            )

            model = sm.OLS(return_y, factors_X)
            results = model.fit()
            # store coefficients, intercept, feature_names, r2, and t-values into self._factor_returns based on the date
            self._factor_returns[date] = {
                "returns": results.params,
                "feature_names": self._fos_risk_factor_uuids,
                "r2": results.rsquared_adj,
                "t_values": results.tvalues,
            }

        self._generate_factor_returns()
        self._generate_idiosyncratic_returns()
        self._generate_factor_covariance()
        self._generate_idiosyncratic_variance()
        print("Done fitting Cross Sectional Regression")
        return

    def _generate_idiosyncratic_returns(self):
        """
        generate idio returns
        """
        print("Generating idiosyncratic returns...")
        # Merge the DataFrames on 'date'
        self._df_idio = pd.merge(
            self._df_loadings,
            self._df_factor_returns.reset_index(),
            on="datetime",
            suffixes=("", "_factor_returns"),
        )

        # Multiplying matching columns
        for col in self._fos_risk_factor_uuids:
            self._df_idio[f"calc_f_r_{col}"] = (
                self._df_idio[col] * self._df_idio[f"returns_{col}"]
            )

        # Dropping the extra columns
        self._df_idio = self._df_idio.drop(columns=self._fos_risk_factor_uuids)

        # drop factor exposure columns
        self._df_idio["factor_return_1d"] = self._df_idio[
            [f"calc_f_r_{col}" for col in (self._fos_risk_factor_uuids)]
        ].sum(axis=1)
        self._df_idio["factor_return_1d_error"] = (
            self._df_idio["factor_return_1d"] - self._df_idio["return_1d"]
        )
        self._df_idio = self._df_idio[
            [
                "datetime",
                "id",
                "return_1d",
                "factor_return_1d",
                "factor_return_1d_error",
            ]
        ]
        self._df_idio_returns = self._df_idio[
            ["datetime", "id", "factor_return_1d_error"]
        ].rename({"factor_return_1d_error": "idio_return"})
        print("Done generating idiosyncratic returns")
        return

    def _generate_factor_analysis_df(self, analysis_col="returns"):
        """
        generate cross sectional regression t_values, r2 and factor_returns
        """
        if analysis_col != "r2":
            list_to_insert = [
                [
                    k,
                    *self._factor_returns[k][analysis_col],
                ]
                for k in self._factor_returns
            ]
            cols_with_analysis = [
                analysis_col + "_" + col for col in self._fos_risk_factor_uuids
            ]
        else:
            list_to_insert = [
                [
                    k,
                    self._factor_returns[k][analysis_col],
                ]
                for k in self._factor_returns
            ]
            cols_with_analysis = [analysis_col]

        return pd.DataFrame(
            list_to_insert,
            columns=[
                "datetime",
                *cols_with_analysis,
            ],
        ).set_index("datetime")

    def compute_regression_r2(self):
        print("Computing regression r2...")
        return self._generate_factor_analysis_df(self, analysis_col="r2")

    def generate_factor_t_values(self):
        print("Generating factor t_values...")
        return self._generate_factor_analysis_df(analysis_col="t_values")

    def _generate_factor_returns(self):
        """Generate factor returns"""
        print("Generating factor returns...")
        self._df_factor_returns = self._generate_factor_analysis_df(
            analysis_col="returns"
        )
        print("Done generating factor returns")
        return

    def _generate_factor_covariance(self):
        """Generate factor covariance"""
        print("Generating factor covariance...")
        if (
            self._adjustments["FactorCovEstArgs"] is not None
        ):  # estimate factor return covariance
            factor_cov_processor = FactorCovarianceProcessor(
                self._df_factor_returns,
                window=self._risk_model_window,
                recalc_freq=self._recalc_freq,
            )
            self._df_cov_raw = factor_cov_processor.est_factor_cov_matrix_raw(
                self._df_factor_returns,
                half_life=self._adjustments["FactorCovEstArgs"]["half_life"],
            )
        print("Done generating factor covariance")
        return

    def _generate_idiosyncratic_variance(self):
        """
        generate idio return variance and apply adjustment
        """
        print("Generating idiosyncratic variance...")
        if "IdioVarEstArgs" in self._adjustments:  # esitmate idiosyncratic variance
            idio_var_processor = IdiosyncraticVarianceProcessor(
                self._df_idio_returns,
                window=self._risk_model_window,
                recalc_freq=self._recalc_freq,
            )
            self._df_idio_var_raw = idio_var_processor.est_ewma_idiosyncratic_variance()
        print("Done Generating idiosyncratic variance")
        return

    def _apply_risk_model_adjustments(self):
        """
        generate factor return covariance and apply covariance matrix adjustment
        """
        # generate raw covariance matrix. This step will happen for sure
        print(
            "Applying Risk Adjustment on Covariance Matrix and Idiosyncratic Variance..."
        )
        if (
            self._adjustments["FactorCovEstArgs"] is not None
        ):  # estimate factor return covariance
            factor_cov_processor = FactorCovarianceProcessor(
                self._df_factor_returns,
                window=self._risk_model_window,
                recalc_freq=self._recalc_freq,
            )
            self._df_cov_raw = factor_cov_processor.est_factor_cov_matrix_raw(
                self._df_factor_returns,
                half_life=self._adjustments["FactorCovEstArgs"]["half_life"],
            )

        if (
            "NewlyWestAdjustmentArgs" in self._adjustments
        ):  # apply newly west covariance adjustment
            self._df_cov_NW = factor_cov_processor.apply_newey_west_adjustment()

        if (
            "EigenfactorRiskAdjustmentArgs" in self._adjustments
        ):  # apply eigenfactor risk ajustment
            self._df_cov_eigen = (
                factor_cov_processor.apply_eigenfactor_risk_adjustment()
            )

        if (
            "FactorVolatilityRegimeAdjustmentArgs" in self._adjustments
        ):  # apply factor volatility risk ajustment
            self._df_cov_VRA = factor_cov_processor.apply_volatility_regime_adjustment()

        print("Done applying risk adjustment")
        return

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

        self.expression = cvx.sum_squares(cvx.multiply(np.sqrt(idiosync_var), w_plus))

        risk_from_factors = factor_load.T @ factor_covar @ factor_load

        self.expression += w_plus @ risk_from_factors @ w_plus.T

        return self.expression, []

    def _drop_excluded_assets(self):
        self.factor_loadings = self._remove_excl_columns(self.factor_loadings)
        self.idiosyncratic_variance = self._remove_excl_columns(
            self.idiosyncratic_variance
        )

    def _remove_excl_columns(self, pd_obj):
        return util.remove_excluded_columns_pd(
            pd_obj,
            exclude_assets=self.exclude_assets,
        )
