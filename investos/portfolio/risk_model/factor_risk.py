import cvxpy as cvx
import numpy as np
import pandas as pd
import numpy as np
import investos.util as util
import time
from investos.portfolio.risk_model.factor_utils import *
from sklearn.linear_model import LinearRegression
from investos.portfolio.risk_model import BaseRisk


class FactorRisk(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self, df_loadings : pd.core.frame.DataFrame,
        regress_col, risk_cols,
        factor_covariance = None, factor_loadings = None, idiosyncratic_variance = None, **kwargs):
        super().__init__(**kwargs)
        self.factor_covariance = factor_covariance
        self.factor_loadings = factor_loadings
        self.idiosyncratic_variance = idiosyncratic_variance

        print("FactorRisk Model")           # test-code
        print(type(df_loadings))
        self._df_loadings = df_loadings
        self._regress_col = regress_col
        self._risk_cols = risk_cols
        self._drop_excluded_assets()

        self._factor_returns = None

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
    
    def _genFactorAndSpecificReturn(self):
        pass

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
    def _genFactorAndSpecificReturn(self, timestamp = 'datetime') -> None:
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
            # model = sm.OLS(y, X).fit()
            # print(model.summary())
            # Print t-values for each factor
            # print("T-Values:")
            # for i, factor in enumerate(X.columns):
            #     print(f"{factor}: {t_values[i]}")
            # Storing the coefficients and intercept
            self._factor_returns[d] = {
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "feature_names": model.feature_names_in_,
                "r2": model.score(X, y),
                "t-values": abs(t_values)
                #"period": df_current.period.unique()
            }

        #self._factor_returns
        return 

    # 
    def run(self):
        TotalStartT = time.perf_counter()
        print("==========Barra Risk Model==========\n1. initialization")
        self._initInfo()
        print(('Time : %.2f' % (time.perf_counter()-TotalStartT, ))+"\n2. Cross-sectional Regression")
        StartT = time.perf_counter()
        self._genFactorAndSpecificReturn()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+"\n3. Estimate Factor Return Covariance Matrix")
        StartT = time.perf_counter()
        #self._genFactorCovariance()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+"\n4. Estimate Idiosyncratic Return Matrix")
        StartT = time.perf_counter()
        #self._genSpecificRisk()
        print("Time : %.2f" % (time.perf_counter()-StartT, )+("\nTotal Time : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        return 0
    
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
