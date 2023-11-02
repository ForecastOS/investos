import datetime as dt

import cvxpy as cvx
import numpy as np
import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.util import remove_excluded_columns_pd, values_in_time


class TradingCost(BaseCost):
    """Calculates per period cost for trades `u`, based on spread, standard deviation, volume, and price.

    Attributes
    ----------
    sensitivity_coeff : float
        For scaling transaction cost; 1 assumes 1 period's volume moves price by 1 std_dev in price
    """

    def __init__(self, forecast_volume, actual_prices, **kwargs):
        super().__init__(**kwargs)
        self.forecast_volume = remove_excluded_columns_pd(
            forecast_volume,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        self.actual_prices = remove_excluded_columns_pd(
            actual_prices,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        # For scaling realized transaction cost; 1 assumes trading 1 day's volume moves price by 1 std dev
        # Look for better research on this and default value
        self.sensitivity_coeff = remove_excluded_columns_pd(
            kwargs.get("price_movement_sensitivity", 1),
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        self.forecast_std_dev = remove_excluded_columns_pd(
            kwargs.get("forecast_std_dev", 0.015),
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        self.half_spread = remove_excluded_columns_pd(
            kwargs.get("half_spread", 5 / 10_000),
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Estimated trading costs.

        Used by optimization strategy to determine trades.
        """
        constraints = []

        std_dev = values_in_time(self.forecast_std_dev, t)
        volume_dollars = values_in_time(self.forecast_volume, t) * values_in_time(
            self.actual_prices, t
        )
        percent_volume_traded_pre_trade_weight = (
            np.abs(value) / volume_dollars
        )  # Multiplied (using cvx) by trade weight (z) below!

        price_movement_term = (
            self.sensitivity_coeff * std_dev * percent_volume_traded_pre_trade_weight
        )

        try:  # Spread (estimated) costs
            self.estimate_expression = cvx.multiply(
                values_in_time(self.half_spread, t), cvx.abs(z)
            )
        except TypeError:
            self.estimate_expression = cvx.multiply(
                values_in_time(self.half_spread, t).values,
                cvx.abs(z),
            )

        try:  # Price movement due to volume (estimated) costs
            self.estimate_expression += cvx.multiply(price_movement_term, cvx.abs(z))
        except TypeError:
            self.estimate_expression += cvx.multiply(
                price_movement_term.values, cvx.abs(z)
            )

        return cvx.sum(self.estimate_expression), constraints

    def get_actual_cost(
        self, t: dt.datetime, h_plus: pd.Series, u: pd.Series
    ) -> pd.Series:
        spread_cost = np.abs(u) * values_in_time(self.half_spread, t)
        std_dev = values_in_time(self.forecast_std_dev, t)
        volume_dollars = values_in_time(self.forecast_volume, t) * values_in_time(
            self.actual_prices, t
        )
        percent_volume_traded = np.abs(u) / volume_dollars

        trading_costs = spread_cost + (
            self.sensitivity_coeff * std_dev * np.abs(u) * percent_volume_traded
        )

        return trading_costs.sum()
