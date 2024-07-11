import cvxpy as cvx
import numpy as np
import pandas as pd
import numpy as np
import investos.util as util
import time

from sklearn.linear_model import LinearRegression
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.risk_model.factor_utils import *
from investos.portfolio.risk_model.factor_covariance_adjustment import *
from investos.portfolio.risk_model.asset_diagonal_variance_adjustment import *

class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self, df_loadings : pd.core.frame.DataFrame,
        regress_col, risk_cols,
        factor_covariance = None, factor_loadings = None, idiosyncratic_variance = None, 
        **kwargs):
        super().__init__(**kwargs)
        self.factor_covariance = factor_covariance
        self.factor_loadings = factor_loadings
        self.idiosyncratic_variance = idiosyncratic_variance

        #print("FactorRisk Model")           # test-code
        #print(type(df_loadings))
        self._df_loadings = df_loadings
        self._regress_col = regress_col
        self._risk_cols = risk_cols
        self._drop_excluded_assets()

        # config for factor return cov
        self.Config_FactorCovESTArgs                     = kwargs.get('FactorCovESTArgs',{'window':21, 'half_life': 360})
        self.Config_NewlyWestAdjustmentArgs              = kwargs.get('NewlyWestAdjustmentArgs', None)
        self.Config_EigenfactorRiskAdjustmentArgs        = kwargs.get('EigenfactorRiskAdjustmentArgs', None)
        self.Config_FactorVolatilityRegimeAdjustmentArgs = kwargs.get('FactorVolatilityRegimeAdjustmentArgs', None)  

        # config for idio return var
        self.Config_IdioVarEstArgs                       = kwargs.get('Config_IdioVarEstArgs',{'window':21, 'half_life': 360})
        
        #print(self.Config_FactorCovESTArgs)
        self._factor_returns = None
        self._df_factor_summary = None
        self._df_r2 = None
        self._df_factor_returns = None
        self._df_factor_t_values = None

        self._df_idio_returns = None

    @property
    def CrossSectionalRegressionSummary(self):
        if not self._df_factor_summary:     # haven't run the cross-sectional regression yet
            raise NotImplementedError("Haven't generated Cross Sectional Regression results")
        return self._df_factor_summary

    @property
    def FactorReturns(self):
        if not self._df_factor_returns:     # haven't run the cross-sectional regression yet
            raise NotImplementedError("Haven't generated Cross Sectional Regression results")
        return self._df_factor_returns

    @property
    def Factor_t_Values(self):
        if not self._df_factor_t_values:     # haven't run the cross-sectional regression yet
            raise NotImplementedError("Haven't generated Cross Sectional Regression results")
        return self._df_factor_t_values

    @property
    def Regression_r2(self):
        if not self._df_r2:     # haven't run the cross-sectional regression yet
            raise NotImplementedError("Haven't generated Cross Sectional Regression results")
        return self._df_r2

    @property
    def DataBase(self) -> pd.core.frame.DataFrame:
        return self._df_loadings
    
    @property
    def regress_df(self) -> pd.core.frame.DataFrame:
        return self._df_loadings[self.regress_col]
    
    @property
    def risk_df(self) -> pd.core.frame.DataFrame:
        return self._df_loadings[self._risk_cols]
    
    @property
    def StartDate(self, datetime_col: str = 'datetime'):
        return self.DataBase.min()
    

    @StartDate.setter
    def startDate(self, start_date: str, datetime_col: str = 'datetime') -> None:
        if self.StartDate > start_date:
            raise ValueError("The current risk data start date or regress data start date is later than the specified start date.")
        self._df_loadings[self._df_loadings[datetime_col] >= start_date]
        return
    

    # Initialization
    def _initInfo(self):
        # if self.RiskESTDTs==[]: raise __QS_Error__("没有设置计算风险数据的时点序列!")
        # FactorNames = set(self.Config.ModelArgs["所有因子"])
        # if not set(self.FT.FactorNames).issuperset(FactorNames): raise __QS_Error__("因子表必须包含如下因子: %s" % FactorNames)
        # self._genRegressDateTime()
        # self._adjustRiskESTDateTime()
        # if self.RiskESTDTs==[]: raise __QS_Error__("可以计算风险数据的时点序列为空!")
        pass
    
    # Cross Sectional Regression
    def _genFactorAndIdioReturn(self, timestamp = 'datetime') -> None:
        """fit a cross sectional factor model and generate factor return covariance and idio returns variance

        Parameters
        ----------
        self._df_loadings : DataFrame
            Linear regression model fits a linear model
        self._risk_cols : list
            the array of X cols (AKA factor columns, or training data columns) 
        self._regress_col : string
            the y col, by default return_1d

        Returns
        -------
        self._factor_returns
            dict:   coefficients, intercept, feature_names, r2, t-values, period
        self._df_factor_summary
            pd.DataFrame: convert self._factor_returns to pd.DataFrame including datetime, r2, factor returns, t-values for corresponding factors
        self._df_r2
            pd.DataFrame: part of the self._df_factor_summary, only contain the r2 values for each of the date
        self._df_factor_returns
            pd.DataFrame: part of the self._df_factor_summary, only contain the factor returns values for each of the date
        self._df_factor_t_value:
            pd.DataFrame: part of the self._df_factor_summary, only contain the t values for factor returns for each of the date
        """


        """
        fit the cross sectional regression
        """
        self._factor_returns = {}
        for d in self._df_loadings[timestamp].unique():
            # Isolating the data for the current date
            df_current = self._df_loadings[self._df_loadings[timestamp] == d]

            # outlier modification and factor standarization
            # identify factor effectiveness
            y = df_current[self._regress_col]
            X = df_current[self._risk_cols]

            # Performing linear regression
            model = LinearRegression(fit_intercept=False).fit(X, y)
            t_values = get_t_statistics(model, X, y)

            # Store coefficients, intercept, feature_names, r2, and t-values into self._factor_returns based on the date
            self._factor_returns[d] = {
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "feature_names": model.feature_names_in_,
                "r2": model.score(X, y),
                "t-values": abs(t_values)
                #"period": df_current.period.unique()
            }
        
        """
        Generate factor returns 
        """
        list_to_insert = [
            [k, 
             self._factor_returns[k]["r2"], 
             *self._factor_returns[k]["t-values"],
             *self._factor_returns[k]["coefficients"]
            ]
             #self._factor_returns[k]["period"][0]]
            for k in self._factor_returns]    # iterative through the factor returns dict

        # split the columns into return, t-value and r2
        cols_with_return = ['returns_' + col for col in self._risk_cols]
        cols_with_t_values = ['t_values_' + col for col in self._risk_cols]
        cols_with_r2 = ['r2']
        # df_factor_returns = df.append(pd.Series(list_to_insert, index=['date', 'r2', *cols]), ignore_index=True)  # using append
        self._df_factor_summary = pd.DataFrame(list_to_insert, columns=["datetime", "r2", *cols_with_t_values, *cols_with_return]).set_index('datetime')


        self._df_r2 = self._df_factor_summary[cols_with_r2]
        self._df_factor_t_values = self._df_factor_summary[cols_with_t_values]
        self._df_factor_returns = self._df_factor_summary[cols_with_return]
        #self._factor_returns

        """
        Generate idio returns
        """
        # Merge the DataFrames on 'date'
        self._df_idio = pd.merge(
            self._df_loadings, self._df_factor_returns.reset_index(), on="datetime", suffixes=("", "_factor_returns")
        )

        # Multiplying matching columns
        for col in self._risk_cols:
            self._df_idio[f"calc_f_r_{col}"] = self._df_idio[col] * self._df_idio[f"returns_{col}"]

        # # Dropping the extra columns
        self._df_idio = self._df_idio.drop(columns=self._risk_cols)  # drop factor exposure columns
        self._df_idio = self._df_idio.drop(columns=cols_with_return)  # drop factor return columns

        self._df_idio["factor_return_1d"] = self._df_idio[[f"calc_f_r_{col}" for col in self._risk_cols]].sum(axis=1)
        self._df_idio["factor_return_1d_error"] = self._df_idio["factor_return_1d"] - self._df_idio["return_1d"]
        self._df_idio = self._df_idio[["datetime", "id", "return_1d", "factor_return_1d", "factor_return_1d_error"]]

        self._df_idio_returns = self._df_idio[["datetime", "id", "factor_return_1d_error"]].rename({"factor_return_1d_error":'idio_return'})
        return 

    def _genFactorCovariance(self):
        """

        """
        Args = {
            "FactorCovESTArgs":                         self.Config_FactorCovESTArgs,
            "NewlyWestAdjustmentArgs":                  self.Config_NewlyWestAdjustmentArgs,
            "EigenfactorRiskAdjustmentArgs":            self.Config_EigenfactorRiskAdjustmentArgs,
            "FactorVolatilityRegimeAdjustmentArgs":     self.Config_FactorVolatilityRegimeAdjustmentArgs
            }
        
        #print(Args)
        """
        generate factor return covariance and covariance matrix adjustment
        """
        # generate raw covariance matrix. This step will happen for sure
        if Args["FactorCovESTArgs"] is not None:
            factorcovadjuster = FactorCovAdjuster(self._df_factor_returns,window = Args["FactorCovESTArgs"]['window']) 
            self._df_cov_raw = factorcovadjuster.calc_fcm_raw(self._df_factor_returns,half_life = Args["FactorCovESTArgs"]['half_life'])
        
        # apply newly west covariance adjustment
        if Args["NewlyWestAdjustmentArgs"] is not None:
            self._df_cov_NW = factorcovadjuster.calc_newey_west_frm(max_lags = Args["NewlyWestAdjustmentArgs"].get('max_lags',1),                      # max_lags default value is 1 
                                                                    multiplier = Args["NewlyWestAdjustmentArgs"].get('multiplier', 1.2),               # multiplier default value is 1.2
                                                                    half_life = Args["NewlyWestAdjustmentArgs"].get('half_life',480))                  # half_life default value is 480

        if Args["EigenfactorRiskAdjustmentArgs"] is not None:
            self._df_cov_eigen = factorcovadjuster.calc_eigenfactor_risk_frm(max_lags = Args["EigenfactorRiskAdjustmentArgs"].get('max_lags',1),       # max_lags default value is 1
                                                                             multiplier = Args["EigenfactorRiskAdjustmentArgs"].get('multiplier', 1.2),# multiplier default value is 1.2
                                                                             half_life = Args["EigenfactorRiskAdjustmentArgs"].get('half_life',480),   # half_life default value is 480 
                                                                             coef = Args["EigenfactorRiskAdjustmentArgs"].get('coef', 1.2),            # coef default value is 1.2               
                                                                             M = Args["EigenfactorRiskAdjustmentArgs"].get('monte_carlo_num', 1000),   # monte carlo simulation number default value is 1.2 
                                                                             window = Args["EigenfactorRiskAdjustmentArgs"].get('window',480)          # window default value is 480
                                                                             )                                                             

        if Args["FactorVolatilityRegimeAdjustmentArgs"] is not None:
            self._df_cov_VRA = factorcovadjuster.calc_eigenfactor_risk_frm(max_lags = Args["FactorVolatilityRegimeAdjustmentArgs"].get('max_lags',1),         # max_lags default value is 1
                                                                           multiplier = Args["FactorVolatilityRegimeAdjustmentArgs"].get('multiplier', 1.2),  # multiplier default value is 1.2
                                                                           half_life = Args["FactorVolatilityRegimeAdjustmentArgs"].get('half_life',480),     # half_life default value is 480 
                                                                           window = Args["FactorVolatilityRegimeAdjustmentArgs"].get('window',480)            # window default value is 480
                                                                           )                                                       

        return 
    

    def _genSpecificRisk(self):

        """

        """
        Args = {
            "IdioVarEstArgs":                           self.Config_IdioVarEstArgs,
            "NewlyWestAdjustmentArgs":                  self.Config_NewlyWestAdjustmentArgs,
            }
        print(Args)
        # generate raw covariance matrix. This step will happen for sure
        if Args["IdioVarEstArgs"] is not None:

            delta = AssetDigonalVarAdjuster(self._df_idio_returns,Args['IdioVarEstArgs']['window'])
            self._df_idio_var_raw = delta.calculate_ewma_idiosyncratic_variance(self._df_idio_returns,half_life = Args["IdioVarEstArgs"]['half_life'])

            
            #self._df_idio_returns
        return

    def run(self) -> None:

        """run the barra risk model
            1. initialization check
            2. generate 

        """

        TotalStartT = time.perf_counter()
        print("==========Barra Risk Model==========\n1. initialization")
        self._initInfo()
        print(('Time : %.2f' % (time.perf_counter()-TotalStartT, ))+"\n2. Cross-sectional Regression")
        StartT = time.perf_counter()
        self._genFactorAndIdioReturn()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+"\n3. Estimate Factor Return Covariance Matrix")
        StartT = time.perf_counter()
        self._genFactorCovariance()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+"\n4. Estimate Idiosyncratic Return Matrix")
        StartT = time.perf_counter()
        self._genSpecificRisk()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+("\nTotal Time : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
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
