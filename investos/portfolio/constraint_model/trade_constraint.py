import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class MaxAbsTurnoverConstraint(BaseConstraint):
    def __init__(self, limit=0.05, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return cvx.sum(cvx.abs(z)) <= self.limit
