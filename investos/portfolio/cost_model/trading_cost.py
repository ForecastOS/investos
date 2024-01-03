import datetime as dt

import cvxpy as cvx
import numpy as np
import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.util import remove_excluded_columns_pd, values_in_time


class TradingCost(BaseCost):
    """Calculates per period cost for trades `u`, based on spread, standard deviation, volume, and price.

    Actual t-cost calculation based on AQR's research
    on market impact for live trades from their execution database
    between 1998 and 2016 (Figure 5):

    Frazzini, Andrea and Israel, Ronen and Moskowitz, Tobias J. and Moskowitz, Tobias J.,
    Trading Costs (April 7, 2018).
    Available at SSRN: https://ssrn.com/abstract=3229719

    Attributes
    ----------
    sensitivity_coeff : float
        For scaling volume-based transaction cost component. Does not impact spread related costs.
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
        self.sensitivity_coeff = (
            remove_excluded_columns_pd(
                kwargs.get("price_movement_sensitivity", 1),
                exclude_assets=self.exclude_assets,
                include_assets=self.include_assets,
            )
            / 100  # Convert to bps
        )
        self.half_spread = remove_excluded_columns_pd(
            kwargs.get("half_spread", 1 / 10_000),
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        self.est_opt_cost_config = kwargs.get(
            "est_opt_cost_config",
            {
                "linear_mi_multiplier": 2,
                "min_half_spread": 8 / 10_000,
            },
        )

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Estimated trading costs.

        Used by optimization strategy to determine trades.
        """
        constraints = []

        volume_dollars = values_in_time(self.forecast_volume, t) * values_in_time(
            self.actual_prices, t
        )
        percent_volume_traded_pre_trade_weight = np.abs(value) / volume_dollars

        try:  # Spread (convex, estimated) costs
            self.estimate_expression = cvx.multiply(
                np.clip(
                    values_in_time(self.half_spread, t),
                    self.est_opt_cost_config["min_half_spread"],
                    None,
                ),
                cvx.abs(z),
            )
        except TypeError:
            self.estimate_expression = cvx.multiply(
                np.clip(
                    values_in_time(self.half_spread, t).values,
                    self.est_opt_cost_config["min_half_spread"],
                    None,
                ),
                cvx.abs(z),
            )

        try:  # Market impact (convex, estimated) costs
            self.estimate_expression += (
                cvx.multiply(
                    cvx.multiply(
                        percent_volume_traded_pre_trade_weight,
                        cvx.abs(z),
                    ),
                    self.sensitivity_coeff,
                )
                * self.est_opt_cost_config["linear_mi_multiplier"]
            )
        except TypeError:
            self.estimate_expression += (
                cvx.multiply(
                    cvx.multiply(
                        percent_volume_traded_pre_trade_weight.values,
                        cvx.abs(z),
                    ),
                    self.sensitivity_coeff,
                )
                * self.est_opt_cost_config["linear_mi_multiplier"]
            )

        return cvx.sum(self.estimate_expression), constraints

    def get_actual_cost(
        self, t: dt.datetime, h_plus: pd.Series, u: pd.Series
    ) -> pd.Series:
        spread_cost = np.abs(u) * values_in_time(self.half_spread, t)
        volume_dollars = values_in_time(self.forecast_volume, t) * values_in_time(
            self.actual_prices, t
        )
        percent_volume_traded = np.abs(u) / volume_dollars

        trading_costs = spread_cost + (
            self.sensitivity_coeff * np.abs(u) * (percent_volume_traded**0.5)
        )

        return trading_costs.sum()
