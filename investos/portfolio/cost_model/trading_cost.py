import datetime as dt

import cvxpy as cvx
import numpy as np
import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.util import get_value_at_t, remove_excluded_columns_pd


class TradingCost(BaseCost):
    """Calculates per period cost for trades, based on spread, standard deviation, volume, and price.

    Actual t-cost calculation approximated (loosely) on AQR's research
    on market impact for live trades from their execution database
    between 1998 and 2016.
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

    def _estimated_cost_for_optimization(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        """Estimated trading costs.

        Used by optimization strategy to determine trades.
        """
        constraints = []

        volume_dollars = get_value_at_t(self.forecast_volume, t) * get_value_at_t(
            self.actual_prices, t
        )
        percent_volume_traded_pre_trade_weight = (
            np.abs(portfolio_value) / volume_dollars
        )

        try:  # Spread (convex, estimated) costs
            self.estimate_expression = cvx.multiply(
                np.clip(
                    get_value_at_t(self.half_spread, t),
                    self.est_opt_cost_config["min_half_spread"],
                    None,
                ),
                cvx.abs(weights_trades),
            )
        except TypeError:
            self.estimate_expression = cvx.multiply(
                np.clip(
                    get_value_at_t(self.half_spread, t).values,
                    self.est_opt_cost_config["min_half_spread"],
                    None,
                ),
                cvx.abs(weights_trades),
            )

        try:  # Market impact (convex, estimated) costs
            self.estimate_expression += (
                cvx.multiply(
                    cvx.multiply(
                        percent_volume_traded_pre_trade_weight,
                        cvx.abs(weights_trades),
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
                        cvx.abs(weights_trades),
                    ),
                    self.sensitivity_coeff,
                )
                * self.est_opt_cost_config["linear_mi_multiplier"]
            )

        return cvx.sum(self.estimate_expression), constraints

    def get_actual_cost(
        self,
        t: dt.datetime,
        dollars_holdings_plus_trades: pd.Series,
        dollars_trades: pd.Series,
    ) -> pd.Series:
        spread_cost = np.abs(dollars_trades) * get_value_at_t(self.half_spread, t)
        volume_dollars = get_value_at_t(self.forecast_volume, t) * get_value_at_t(
            self.actual_prices, t
        )
        percent_volume_traded = np.abs(dollars_trades) / volume_dollars

        trading_costs = spread_cost + (
            self.sensitivity_coeff
            * np.abs(dollars_trades)
            * (percent_volume_traded**0.5)
        )

        return trading_costs.sum()
