import cvxpy as cvx
import numpy as np
import pandas as pd
import numpy as np
import statsmodels.api as sm
import investos.util as util
import time
import os
import forecastos as fos # from pypi
from functools import reduce
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.risk_model.factor_utils import *
from investos.portfolio.risk_model.factor_covariance_adjustment import *
from investos.portfolio.risk_model.asset_diagonal_variance_adjustment import *

# will comment out this later
class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self, start_date,end_date, risk_model_window: int,  recalc_freq: int, fos_risk_factor_uuids: list = [], fos_return_uuid: list = ["return_1d"],
        adjustments: dict = {'FactorCovESTArgs':{'window':21, 'half_life': 360}},
        factor_covariance = None, factor_loadings = None, idiosyncratic_variance = None,
        **kwargs):
        super().__init__(**kwargs)
        self._risk_model_window = risk_model_window             #pd.Timedelta(days = risk_model_window)
        self._recalc_freq = recalc_freq                         #pd.Timedelta(days = recalc_freq)

        self._fos_risk_factor_uuids = fos_risk_factor_uuids
        self._fos_return_uuid = fos_return_uuid

        self._end_date = end_date
        self._start_date = start_date
        self._adjustments = adjustments
        #self._drop_excluded_assets()

        if self._fos_risk_factor_uuids: 
            print("=================Barra Risk Model================= ")
            print("\n1. Initialization")
            self._pull_fos_risk_factor_dfs()
            print("\n2. Cross Sectional Regression")
            self._generate_risk_models()
            print("\n3. Apply Risk Adjustment")
            self._apply_risk_model_adjustments()
        else:
            self._factor_covariance = factor_covariance
            self._factor_loadings = factor_loadings
            self._idiosyncratic_variance = idiosyncratic_variance
        #self._drop_excluded_assets()

    # Initialization
    def _pull_fos_risk_factor_dfs(self):
        # pull from feature hub
        fos.api_key = os.environ.get("FORECASTOS_API_KEY")                  #.env including my api_key
        if not fos.api_key:
            raise ValueError("Feature hub connection failed. Please check the API key is valid.")
        dataframes = [fos.Feature.get(factor_uuids[factor]).get_df().rename(columns={'value': factor})
                       for factor in (self._fos_risk_factor_uuids + self._fos_return_uuid)  if factor != 'rbics']

        if dataframes:
            df_merged = reduce(lambda left, right: pd.merge(left, right, how='left', on=['datetime', 'id']), dataframes)
        else:
            raise ValueError("factor loading is empty. Please check the risk factor uuids")
        # special handling industry factors
        if 'rbics' in self._fos_risk_factor_uuids:
            df_industry = pd.get_dummies(fos.Feature.get(factor_uuids['rbics']).get_df().rename(columns={'value': 'industry'}),columns = ['industry'])
            df_industry.iloc[:,1:] = df_industry.iloc[:,1:].astype(int)     # convert bool to int for industry exposure
            df_merged = df_merged.merge(df_industry,how='left', on='id')

            self._fos_risk_factor_uuids.remove('rbics')
            self._fos_industry_factors = [col for col in df_industry.columns if col.startswith('industry_')]
        
        self._df_loadings =df_merged.replace([np.inf, -np.inf], np.nan).dropna()
        self._df_loadings = self._df_loadings[(self._df_loadings.datetime >= self._end_date) & (self._df_loadings.datetime <= self._start_date)]
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
        scaler = StandardScaler()
        self._factor_returns = {}
        for d in sorted(self._df_loadings[timestamp].unique()):                         # iterate through everyday
            # Isolating the data for the current date
            df_current = self._df_loadings[self._df_loadings[timestamp] == d]

            # outlier modification and factor standarization
            # identify factor effectiveness
            for col in self._fos_risk_factor_uuids:
                df_current.loc[:,col] = mstats.winsorize(df_current.loc[:,col], limits=(0.05, 0.95))
                df_current.loc[:,col] = scaler.fit_transform(df_current.loc[:,col].values.reshape(-1, 1)).flatten()
            
            for col in self._fos_return_uuid:
                df_current.loc[:,col] = mstats.winsorize(df_current.loc[:,col], limits=(0.05, 0.95))

            y,X = df_current.loc[:,self._fos_return_uuid ], df_current.loc[:,(self._fos_risk_factor_uuids + self._fos_industry_factors)]
            X.loc[:,'const'] = 1    # need the intercept column, will convert this to country factor later
            model = sm.OLS(y, X)
            results = model.fit()
            # Store coefficients, intercept, feature_names, r2, and t-values into self._factor_returns based on the date
            self._factor_returns[d] = {"coefficients": results.params,
                                       "intercept":  results.params['const'],  # the intercept needs the X col of all 1
                                       "feature_names": self._fos_risk_factor_uuids + self._fos_industry_factors,
                                       "r2": results.rsquared_adj,
                                       "t-values": results.tvalues}
        """
        Generate factor returns 
        """
        list_to_insert = [[k, self._factor_returns[k]["r2"], *self._factor_returns[k]["t-values"][:-1],*self._factor_returns[k]["coefficients"][:-1]]for k in self._factor_returns]    # iterative through the factor returns dict

        # split the columns into return, t-value and r2
        cols_with_return = ['returns_' + col for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]
        cols_with_t_values = ['t_values_' + col for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]
        self._df_factor_summary = pd.DataFrame(list_to_insert, columns=["datetime", "r2", *cols_with_t_values, *cols_with_return]).set_index('datetime')
        self._df_factor_returns = self._df_factor_summary[cols_with_return]
        """
        Generate idio returns
        """
        # Merge the DataFrames on 'date'
        self._df_idio = pd.merge(self._df_loadings, self._df_factor_returns.reset_index(), on="datetime", suffixes=("", "_factor_returns"))

        # Multiplying matching columns
        for col in (self._fos_risk_factor_uuids + self._fos_industry_factors):
            self._df_idio[f"calc_f_r_{col}"] = self._df_idio[col] * self._df_idio[f"returns_{col}"]

        # # Dropping the extra columns
        self._df_idio = self._df_idio.drop(columns=self._fos_risk_factor_uuids).drop(columns = self._fos_industry_factors).drop(columns=cols_with_return)   # drop factor exposure columns
        self._df_idio["factor_return_1d"] = self._df_idio[[f"calc_f_r_{col}" for col in (self._fos_risk_factor_uuids + self._fos_industry_factors)]].sum(axis=1)
        self._df_idio["factor_return_1d_error"] = self._df_idio["factor_return_1d"] - self._df_idio["return_1d"]
        self._df_idio = self._df_idio[["datetime", "id", "return_1d", "factor_return_1d", "factor_return_1d_error"]]
        self._df_idio_returns = self._df_idio[["datetime", "id", "factor_return_1d_error"]].rename({"factor_return_1d_error":'idio_return'})
        return 

    def _apply_risk_model_adjustments(self):
        """
        generate factor return covariance and apply covariance matrix adjustment
        """
        # generate raw covariance matrix. This step will happen for sure
        if self._adjustments["FactorCovESTArgs"] is not None:
            factorcovadjuster = FactorCovAdjuster(self._df_factor_returns,window = self._risk_model_window, recalc_freq= self._recalc_freq) 
            self._df_cov_raw = factorcovadjuster.calc_fcm_raw(self._df_factor_returns,half_life = self._adjustments["FactorCovESTArgs"]['half_life'])
        
        # apply newly west covariance adjustment
        if "NewlyWestAdjustmentArgs" in self._adjustments:
            print("\n apply Newly-West Adjustment")
            self._df_cov_NW = factorcovadjuster.calc_newey_west_frm()                  

        if "EigenfactorRiskAdjustmentArgs" in self._adjustments:
            print("\n apply Eigenfactor Risk Adjustment")
            self._df_cov_eigen = factorcovadjuster.calc_eigenfactor_risk_frm()                                                             

        if "FactorVolatilityRegimeAdjustmentArgs" in self._adjustments:
            print("\n apply Factor Volatility Regime Adjustment")
            self._df_cov_VRA = factorcovadjuster.calc_volatility_regime_frm()                                                       

        """
        generate idio return variance and apply adjustment
        """
        if "IdioVarEstArgs" in self._adjustments:
            print("\n Estimate Idio Return Variance")
            delta = AssetDigonalVarAdjuster(self._df_idio_returns,window = self._risk_model_window, recalc_freq=self._recalc_freq)
            self._df_idio_var_raw = delta.calculate_ewma_idiosyncratic_variance()             #self._df_idio_returns
        return 

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated costs for backtest performance.
        """

        # TODO: need to update this
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
