import cvxpy as cvx
import numpy as np
import pandas as pd

import investos.util as util
from investos.portfolio.risk_model import BaseRisk


class StatFactorRisk(BaseRisk):
    """PCA-factor based risk model.

    The only requirement of custom risk models is that they implement a `_estimated_cost_for_optimization` method.

    Note: risk models are like cost models, except they return 0 for their `value_expr` method (because they only influence optimization weights, not actual cash costs).
    """

    def __init__(self, actual_returns: pd.DataFrame, n_factors=5, **kwargs):
        super().__init__(**kwargs)
        self.n = n_factors
        self.actual_returns = util.remove_excluded_columns_pd(
            actual_returns,
            exclude_assets=self.exclude_assets,
        )
        self.start_date = kwargs.get("start_date", self.actual_returns.index[0])
        self.end_date = kwargs.get("end_date", self.actual_returns.index[-1])
        self.recalc_each_i_periods = kwargs.get("recalc_each_i_periods", False)
        self.timedelta = kwargs.get("timedelta", pd.Timedelta("730 days"))

        self.factor_variance = kwargs.get("factor_variance", None)
        self.factor_loadings = kwargs.get("factor_loadings", None)
        self.idiosyncratic_variance = kwargs.get("idiosyncratic_variance", None)

        if kwargs.get("calc_risk_model_on_init", False):
            self.create_risk_model(t=self.start_date)

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated costs for backtest performance.
        """
        if (
            self.factor_variance is None
            or self.factor_loadings is None
            or self.idiosyncratic_variance is None
            or (
                self.recalc_each_i_periods
                and self.actual_returns.index.get_loc(t) % self.recalc_each_i_periods
                == 0
            )
        ):
            self.create_risk_model(t=t)

        self.expression = cvx.sum_squares(
            cvx.multiply(np.sqrt(self.idiosyncratic_variance), w_plus)
        )

        risk_from_factors = (self.factor_loadings @ np.sqrt(self.factor_variance)).T

        self.expression += cvx.sum_squares(w_plus @ risk_from_factors)

        return self.expression, []

    def create_risk_model(self, t):
        df = self.actual_returns
        df = df[(df.index < t) & (df.index >= pd.to_datetime(t) - self.timedelta)]

        covariance_matrix = df.cov().dropna().values
        eigenvalue, eigenvector = np.linalg.eigh(covariance_matrix)

        self.factor_variance = eigenvalue[-self.n :]

        self.factor_loadings = pd.DataFrame(
            data=eigenvector[:, -self.n :], index=df.columns
        )
        self.idiosyncratic_variance = pd.Series(
            data=np.diag(
                eigenvector[:, : -self.n]
                @ np.diag(eigenvalue[: -self.n])
                @ eigenvector[:, : -self.n].T
            ),
            index=df.columns,
        )

        self._drop_excluded_assets()

    def _drop_excluded_assets(self):
        self.factor_loadings = util.remove_excluded_columns_pd(
            self.factor_loadings,
            exclude_assets=self.exclude_assets,
        )
        self.idiosyncratic_variance = util.remove_excluded_columns_pd(
            self.idiosyncratic_variance,
            exclude_assets=self.exclude_assets,
        )
