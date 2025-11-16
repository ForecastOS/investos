from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class MaxWeightConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the weight of each asset in a portfolio.

    Parameters
    ----------
    limit : float, optional
        The maximum weight of each asset in the portfolio. Defaults to 0.025.

    **kwargs :
        Additional keyword arguments.
    """

    def __init__(self, limit: float = 0.025, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_portfolio_plus_trades <= self.limit


class MinWeightConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the weight of each asset in a portfolio.

    Parameters
    ----------
    limit : float, optional
        The minimum weight of each asset in the portfolio. Defaults to -0.025.

    **kwargs :
        Additional keyword arguments.
    """

    def __init__(self, limit: float = -0.025, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_portfolio_plus_trades >= self.limit


class ZeroWeightConstraint(BaseConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_portfolio_plus_trades == 0


class ZeroTradeWeightConstraint(BaseConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_trades == 0


class MaxTradeWeightConstraint(BaseConstraint):
    def __init__(self, limit=0.05, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_trades <= self.limit


class MinTradeWeightConstraint(BaseConstraint):
    def __init__(self, limit=-0.03, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_trades >= self.limit
