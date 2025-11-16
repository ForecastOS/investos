import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class MaxLeverageConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the (absolute) leverage of the portfolio.

    E.g. For leverage of 2.0x, a portfolio with 100MM net value
    (i.e. the portfolio value if it were converted into cash,
    ignoring liquidation / trading costs)
    could have 200MM of (combined long and short) exposure.
    """

    def __init__(self, limit: float = 1.0, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.abs(weights_portfolio_plus_trades)) <= self.limit


class MaxShortLeverageConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the short leverage of the portfolio.
    """

    def __init__(self, limit: float = 1.0, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.abs(cvx.neg(weights_portfolio_plus_trades))) <= self.limit


class MaxLongLeverageConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the long leverage of the portfolio.
    """

    def __init__(self, limit: float = 1.0, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.pos(weights_portfolio_plus_trades)) <= self.limit


class MaxLongTradeLeverageConstraint(BaseConstraint):
    def __init__(self, limit=0.025, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.abs(cvx.pos(weights_trades))) <= self.limit


class MaxShortTradeLeverageConstraint(BaseConstraint):
    def __init__(self, limit=0.025, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return cvx.sum(cvx.abs(cvx.neg(weights_trades))) <= self.limit
