from investos.portfolio.constraint_model.base_constraint import BaseConstraint


class MaxWeightConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the weight of each asset in a portfolio.

    Parameters
    ----------
    limit : float, optional
        The maximum weight of each asset in the portfolio. Defaults to 0.05.

    **kwargs :
        Additional keyword arguments.

    Methods
    -------
    weight_expr(self, t, w_plus, z, v):
        Returns a series of holding constraints based on the portfolio weights after trades.

    """

    def __init__(self, limit: float = 0.025, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """
        Returns a series of holding constraints.

        Parameters
        ----------
        t : datetime
            The current time.

        w_plus : series
            Portfolio weights after trades z.

        z : series
            Trades for period t

        v : float
            Value of portfolio at period t

        Returns
        -------
        series
            The holding constraints based on the portfolio weights after trades.
        """
        return w_plus <= self.limit


class MinWeightConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the weight of each asset in a portfolio.

    Parameters
    ----------
    limit : float, optional
        The minimum weight of each asset in the portfolio. Defaults to -0.05.

    **kwargs :
        Additional keyword arguments.

    Methods
    -------
    weight_expr(self, t, w_plus, z, v):
        Returns a series of holding constraints based on the portfolio weights after trades.

    """

    def __init__(self, limit: float = -0.025, exclude_assets=["cash"], **kwargs):
        super().__init__(exclude_assets=exclude_assets, **kwargs)
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """
        Returns a series of holding constraints.

        Parameters
        ----------
        t : datetime
            The current time.

        w_plus : series
            Portfolio weights after trades z.

        z : series
            Trades for period t

        v : float
            Value of portfolio at period t

        Returns
        -------
        series
            The holding constraints based on the portfolio weights after trades.
        """
        return w_plus >= self.limit


class ZeroWeightConstraint(BaseConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return w_plus == 0


class ZeroTradeWeightConstraint(BaseConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return z == 0


class MaxTradeWeightConstraint(BaseConstraint):
    def __init__(self, limit=0.05, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return z <= self.limit


class MinTradeWeightConstraint(BaseConstraint):
    def __init__(self, limit=-0.03, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return z >= self.limit
