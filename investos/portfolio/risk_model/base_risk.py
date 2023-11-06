import datetime as dt

import pandas as pd

from investos.portfolio.cost_model import BaseCost


class BaseRisk(BaseCost):
    """Base risk model for InvestOS.

    The only requirement of custom risk models is that they implement a `_estimated_cost_for_optimization` method.

    Note: risk models are like cost models, except they return 0 for their `value_expr` method (because they only influence optimization weights, not actual cash costs).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
        """Method that calculates per-period costs given period `t` holdings and trades.

        ALWAYS 0 FOR RISK MODELS; DO NOT OVERRIDE. RISK DOESN'T HAVE A CASH COST, IT ONLY AFFECTS OPTIMIZED ASSET WEIGHTS GIVEN `_estimated_cost_for_optimization` RISK (UTILITY) PENALTY.
        """
        return 0

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated costs for backtest performance.
        """
        raise NotImplementedError
