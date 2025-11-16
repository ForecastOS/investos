import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint
from investos.util import get_value_at_t, remove_excluded_columns_pd


class TradeReturnConstraint(BaseConstraint):
    def __init__(self, forecast_returns, costs=[], limit=0.01, **kwargs):
        self.costs = costs
        self.limit = limit
        super().__init__(**kwargs)
        self.forecast_returns = remove_excluded_columns_pd(
            forecast_returns, self.exclude_assets
        )
        self.forecast_returns -= self.limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        sum_forecast_alpha = cvx.sum(
            cvx.multiply(
                get_value_at_t(self.forecast_returns, t).values, weights_trades
            )
        )

        costs_li = []
        for cost in self.costs:
            # Trade specific, not portfolio level (hence 2 weights_trades and no weights_portfolio_plus_trades)
            cost_expr, _ = cost.cvxpy_expression(
                t,
                weights_trades,
                weights_trades,
                portfolio_value,
                self.forecast_returns.index,
            )
            costs_li.append(cost_expr)

        return (sum_forecast_alpha - cvx.sum(costs_li)) >= 0


class TradeGrossReturnConstraint(BaseConstraint):
    def __init__(self, forecast_returns, limit=0.01, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)
        self.forecast_returns = remove_excluded_columns_pd(
            forecast_returns, self.exclude_assets
        )

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        sum_forecast_alpha = cvx.sum(
            cvx.multiply(
                get_value_at_t(self.forecast_returns, t).values, weights_trades
            )
        )

        return sum_forecast_alpha <= self.limit
