import cvxpy as cvx
import numpy as np
import pandas as pd
import numpy as np
import statsmodels.api as sm
import investos.util as util
import os
import forecastos as fos # from pypi
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.risk_model.utils.risk_adjustments import *

global factor_uuids
factor_uuids = {
    # Industry
    "industry": "6f8829e2-6424-4c9a-b05d-d0e5c580859b",

    # For momentum and market beta
    "return_1d": "ea4d2557-7f8f-476b-b4d3-55917a941bb5",

    # Size
    "market_cap_open_dil": "dfa7e6a3-671d-41b2-89e3-10b7bdcf7af9",

    # For liquidity (probably very correlated with size and not needed)
    "volume": "44fa6e70-dd84-4be4-90a4-cb0f2b62c0ca", # Daily volume traded, split adjusted, in thousands of shares. ID is FactSet Sym ID.
    "price_open": "dc41d72a-ba23-4ab2-90ff-6ab776f7a16b", # Share price at open, split adjusted. ID is FactSet Sym ID.

    # Leverage
    "debt_total_prev_1q_to_ebit_ltm": "53a422bf-1dab-4d1e-a9a7-2478a226435b",
    "debt_total_to_market_cap_open_dil_prev_1q": "d862bc25-e085-442a-acfe-05c880da2d40", # 'prev_1q debt_total to market_cap_open_dil multiple. ID is FactSet Sym ID.'

    # Quality
    "net_income_div_sales_ltm": "7f1e058f-46b6-406f-81a1-d8a5b81371a2", # ltm profit margin (as % of sales)
    "ebit_div_sales_ltm": "47d7d1b0-46cc-4620-b0d3-35aa33fe490a",

    # Value
    "ev_to_ebit_ltm": "f65f7b6a-5433-4543-8f15-5f50e43dd9f9",
    "market_cap_open_dil_to_net_income_ltm": "7e4264a9-a137-4f5b-bc10-1f1e90d2f3f5",
    "market_cap_open_dil_to_operating_excl_wc_cf_ltm": "5f050fce-5099-4ce9-a737-5d8f536c5826", # ltm market_cap_open_dil to operating_excl_wc_cf multiple. ID is FactSet Sym ID.

    # Growth
    "sales_ltm_growth_over_sales_ltm_lag_1a": "e6035b0a-65b9-409d-a02e-b3d47ca422e2",
}

class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self, start_date,end_date, risk_model_window: int,  recalc_freq: int, fos_risk_factor_uuids: list = [], fos_return_uuid: list = ["return_1d"],
        adjustments: dict = {'FactorCovEstArgs':{'window':21, 'half_life': 360}},
        factor_covariance = None, factor_loadings = None, idiosyncratic_variance = None,
        **kwargs):
        super().__init__(**kwargs)

        self._risk_model_window = risk_model_window
        self._recalc_freq = recalc_freq

        self._fos_risk_factor_uuids = fos_risk_factor_uuids
        self._fos_return_uuid = fos_return_uuid

        self._end_date = end_date
        self._start_date = start_date
        self._adjustments = adjustments

        if self._fos_risk_factor_uuids:
            print("Running Barra Risk model...")
            self._pull_fos_risk_factor_dfs()
            self._generate_risk_models()
            self._apply_risk_model_adjustments()
            print("\nDone generating factor loadings,factor covariance, and idiosyncratic variance")
        else:
            self._factor_covariance = factor_covariance
            self._factor_loadings = factor_loadings
            self._idiosyncratic_variance = idiosyncratic_variance



    # Initialization
    def _pull_fos_risk_factor_dfs(self):
        # pull from feature hub
        print("\n1. Pulling out data from FeatureHub...")

        if not fos.api_key:
            raise ValueError("Feature hub connection failed. Please check the API key is valid.")

        # Initialize a DataFrame
        self._df_loadings = fos.Feature.get(factor_uuids[self._fos_return_uuid[0]]).get_df().rename(columns={'value':self._fos_return_uuid[0]})

        for factor in self._fos_risk_factor_uuids:
            temp_df = fos.Feature.get(factor_uuids[factor]).get_df().rename(columns={'value': factor})
            if 'datetime' not in temp_df.columns:
                # Perform operations for non-datetime columns (assuming to create dummies)
                temp_df = pd.get_dummies(temp_df, columns=[factor], dtype=int)
                self._df_loadings = self._df_loadings.merge(temp_df, how='left', on='id')
                self._fos_risk_factor_uuids.remove(factor)
            else:
                self._df_loadings = pd.merge_asof(self._df_loadings.sort_values('datetime'),temp_df.sort_values('datetime'),on='datetime',by='id',direction='forward')
        # handling missing data and inf data
        self._df_loadings = self._df_loadings.replace([np.inf, -np.inf], np.nan).dropna()
        self._df_loadings = self._df_loadings[(self._df_loadings.datetime >= self._end_date) & (self._df_loadings.datetime <= self._start_date)]

        print("Done pulling data")
        return

    def _generate_risk_models(self, timestamp = 'datetime') -> None:
        """fit a cross sectional factor model and generate factor return covariance and idio returns variance

        Parameters
        ----------
        self._df_loadings : DataFrame
            Linear regression model fits a linear model
        self._fos_risk_factor_uuids : list
            the array of X cols (AKA factor columns, or training data columns)
        self._regress_col : string
            the y col, by default return_1d

        Returns
        -------
        """

        """
        fit the cross sectional regression
        """
        print("\n2. Running Cross Sectional Regression...")
        self._fos_industry_factors = [col for col in self._df_loadings.columns if col not in ['datetime', 'id'] + self._fos_risk_factor_uuids + self._fos_return_uuid]
        scaler = StandardScaler()
        self._factor_returns = {}
        for date in sorted(self._df_loadings[timestamp].unique()):
            # isolating the data for the current date
            df_current = self._df_loadings[self._df_loadings[timestamp] == date]

            # winsorzie and standarize risk factors
            for col in self._fos_risk_factor_uuids:
                df_current.loc[:,col] = mstats.winsorize(df_current.loc[:,col], limits=(0.05, 0.95))
                df_current.loc[:,col] = scaler.fit_transform(df_current.loc[:,col].values.reshape(-1, 1)).flatten()
            # winsorzie return
            for col in self._fos_return_uuid:
                df_current.loc[:,col] = mstats.winsorize(df_current.loc[:,col], limits=(0.05, 0.95))

            y,X = df_current.loc[:,self._fos_return_uuid ], df_current.loc[:,(self._fos_risk_factor_uuids + self._fos_industry_factors)]
            X.loc[:,'const'] = 1    # need the intercept column, will convert this to country factor later
            model = sm.OLS(y, X)
            results = model.fit()
            # store coefficients, intercept, feature_names, r2, and t-values into self._factor_returns based on the date
            self._factor_returns[date] = {"coefficients": results.params,
                                       "intercept":  results.params['const'],  # the intercept needs the X col of all 1
                                       "feature_names": self._fos_risk_factor_uuids + self._fos_industry_factors,
                                       "r2": results.rsquared_adj,
                                       "t-values": results.tvalues}

        self._generate_factor_returns()
        self._generate_idiosyncratic_returns()
        print("Done regression")
        return

    def _generate_idiosyncratic_returns(self):
        """
        Generate idio returns
        """
        print("\n generating idiosyncratic returns...")
        # Merge the DataFrames on 'date'
        self._df_idio = pd.merge(self._df_loadings, self._df_factor_returns.reset_index(), on="datetime", suffixes=("", "_factor_returns"))

        # Multiplying matching columns
        for col in (self._fos_risk_factor_uuids + self._fos_industry_factors):
            self._df_idio[f"calc_f_r_{col}"] = self._df_idio[col] * self._df_idio[f"returns_{col}"]

        # Dropping the extra columns
        self._df_idio = self._df_idio.drop(columns=self._fos_risk_factor_uuids).drop(columns = self._fos_industry_factors).drop(columns=self.cols_with_return)   # drop factor exposure columns
        self._df_idio["factor_return_1d"] = self._df_idio[[f"calc_f_r_{col}" for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]].sum(axis=1)
        self._df_idio["factor_return_1d_error"] = self._df_idio["factor_return_1d"] - self._df_idio["return_1d"]
        self._df_idio = self._df_idio[["datetime", "id", "return_1d", "factor_return_1d", "factor_return_1d_error"]]
        self._df_idio_returns = self._df_idio[["datetime", "id", "factor_return_1d_error"]].rename({"factor_return_1d_error":'idio_return'})
        return

    def _generate_factor_returns(self):
        """
        Generate factor returns
        """
        print("\n generating factor returns...")
        list_to_insert = [[k, self._factor_returns[k]["r2"], *self._factor_returns[k]["t-values"][:-1],*self._factor_returns[k]["coefficients"][:-1]]for k in self._factor_returns]    # iterative through the factor returns dict

        # split the columns into return, t-value and r2
        self.cols_with_return = ['returns_' + col for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]
        self.cols_with_t_values = ['t_values_' + col for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]
        self._df_factor_summary = pd.DataFrame(list_to_insert, columns=["datetime", "r2", *self.cols_with_t_values, *self.cols_with_return]).set_index('datetime')
        self._df_factor_returns = self._df_factor_summary[self.cols_with_return]
        return

    def _apply_risk_model_adjustments(self):
        """
        generate factor return covariance and apply covariance matrix adjustment
        """
        # generate raw covariance matrix. This step will happen for sure
        print("\n3. Applying Risk Adjustment on Covariance Matrix and Idiosyncratic Variance...")
        if self._adjustments["FactorCovEstArgs"] is not None:               # estimate factor return covariance
            print("\n Estimating Factor Covariance")
            factorcovadjuster = FactorCovAdjuster(self._df_factor_returns,window = self._risk_model_window, recalc_freq= self._recalc_freq)
            self._df_cov_raw = factorcovadjuster.est_factor_cov_matrix_raw(self._df_factor_returns,half_life = self._adjustments["FactorCovEstArgs"]['half_life'])

        if "NewlyWestAdjustmentArgs" in self._adjustments:                  # apply newly west covariance adjustment
            print("\n Applying Newly-West Adjustment")
            self._df_cov_NW = factorcovadjuster.apply_newey_west_adjustment()

        if "EigenfactorRiskAdjustmentArgs" in self._adjustments:            # apply eigenfactor risk ajustment
            print("\n Applying Eigenfactor Risk Adjustment")
            self._df_cov_eigen = factorcovadjuster.apply_eigenfactor_risk_adjustment()

        if "FactorVolatilityRegimeAdjustmentArgs" in self._adjustments:     # apply factor volatility risk ajustment
            print("\n Applying Factor Volatility Regime Adjustment")
            self._df_cov_VRA = factorcovadjuster.apply_volatility_regime_adjustment()

        """
        generate idio return variance and apply adjustment
        """

        if "IdioVarEstArgs" in self._adjustments:                           # esitmate idiosyncratic variance
            print("\n Estimating Idiosyncratic Return Variance")
            delta = AssetDigonalVarAdjuster(self._df_idio_returns,window = self._risk_model_window, recalc_freq=self._recalc_freq)
            self._df_idio_var_raw = delta.est_ewma_idiosyncratic_variance()

        print("Done risk adjustment")
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
