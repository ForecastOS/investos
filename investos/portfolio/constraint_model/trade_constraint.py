import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class MaxAbsTurnoverConstraint(BaseConstraint):
    def __init__(self, limit=0.05, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.abs(weights_trades)) <= self.limit
