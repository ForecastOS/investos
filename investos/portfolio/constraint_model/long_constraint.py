from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class LongOnlyConstraint(BaseConstraint):
    """
    A constraint that enforces no short positions. Including no short cash position.

    Parameters
    ----------
    **kwargs :
        Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_portfolio_plus_trades >= 0.0


class LongCashConstraint(BaseConstraint):
    """
    A constraint that enforces no short cash positions.

    Parameters
    ----------
    **kwargs :
        Additional keyword arguments.
    """

    def __init__(self, include_assets=["cash"], **kwargs):
        super().__init__(include_assets=include_assets, **kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return weights_portfolio_plus_trades >= 0.0


class EqualLongShortConstraint(BaseConstraint):
    """
    A constraint that enforces equal long and short exposure.

    Parameters
    ----------
    **kwargs :
        Additional keyword arguments.
    """

    def __init__(self, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return sum(weights_portfolio_plus_trades) == 0.0


class EqualLongShortTradeConstraint(BaseConstraint):
    def __init__(self, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return sum(weights_trades) == 0.0
