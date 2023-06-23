import pandas as pd
import numpy as np
import cvxpy as cvx

from investos.portfolio.risk_model import BaseRisk

class StatFactorRisk(BaseRisk):
    """PCA-factor based risk model.

    The only requirement of custom risk models is that they implement a `_estimated_cost_for_optimization` method.

    Note: risk models are like cost models, except they return 0 for their `value_expr` method (because they only influence optimization weights, not actual cash costs).
    """

    def __init__(self, n_factors=5):
        self.n = n_factors

        self.optimizer = None # Set during Controller initialization

        self.factor_variance = None
        self.factor_loadings = None
        self.idiosyncratic_variance = None

        super().__init__()


    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades. 

        Not used to calculate simulated costs for backtest performance.
        """
        if self.factor_variance is None or self.factor_loadings is None or self.idiosyncratic_variance is None:
            self.create_risk_model()

        self.expression = cvx.sum(
            cvx.multiply(
                self.idiosyncratic_variance, 
                cvx.square(w_plus)
            )
        ) + cvx.sum(
            cvx.multiply(
                w_plus.T @ self.factor_loadings,
                self.factor_variance
            )
        )
        
        return self.expression, []


    def create_risk_model(self):
        df = self.optimizer.actual['return']
        forecast_start_d = self.optimizer.forecast['date']['start']
        df = df[(
            df.index < forecast_start_d) & (
            df.index >= pd.to_datetime(forecast_start_d) - pd.Timedelta("730 days") # 2 years
        )]

        covariance_matrix = df.cov().dropna().values
        eigenvalue, eigenvector = np.linalg.eigh(covariance_matrix)
        
        self.factor_variance = eigenvalue[-self.n:]
        self.factor_loadings = pd.DataFrame(
            data=eigenvector[:,-self.n:], 
            index=df.columns
        )
        self.factor_loadings.iloc[-1] = 0 # For cash
        self.idiosyncratic_variance = pd.Series(
            data=np.diag(
                eigenvector[:, :-self.n] @ 
                np.diag(
                    eigenvalue[:-self.n]
                ) @ eigenvector[:, :-self.n].T
            ),
            index=df.columns
        )
        self.idiosyncratic_variance = self.idiosyncratic_variance.copy()
        self.idiosyncratic_variance.iloc[-1] = 0