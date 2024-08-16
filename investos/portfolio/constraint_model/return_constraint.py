import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint
from investos.util import remove_excluded_columns_pd, values_in_time


class TradeReturnConstraint(BaseConstraint):
    def __init__(self, forecast_returns, costs=[], limit=0.01, **kwargs):
        self.costs = costs
        self.limit = limit
        super().__init__(**kwargs)
        self.forecast_returns = remove_excluded_columns_pd(
            forecast_returns, self.exclude_assets
        )
        self.forecast_returns -= self.limit

    def _weight_expr(self, t, w_plus, z, v):
        alpha_term = cvx.sum(
            cvx.multiply(values_in_time(self.forecast_returns, t).values, z)
        )

        costs_li = []
        for cost in self.costs:
            # Trade specific, not portfolio level (hence 2 z's and no w_plus)
            cost_expr, _ = cost.weight_expr(t, z, z, v, self.forecast_returns.index)
            costs_li.append(cost_expr)

        return (alpha_term - cvx.sum(costs_li)) >= 0


class TradeGrossReturnConstraint(BaseConstraint):
    def __init__(self, forecast_returns, limit=0.01, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)
        self.forecast_returns = remove_excluded_columns_pd(
            forecast_returns, self.exclude_assets
        )

    def _weight_expr(self, t, w_plus, z, v):
        sum_forecast_alpha = cvx.sum(
            cvx.multiply(values_in_time(self.forecast_returns, t).values, z)
        )

        return sum_forecast_alpha <= self.limit
