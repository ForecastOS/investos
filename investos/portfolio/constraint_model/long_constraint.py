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
            The holding constraints based on the no short positions (including no short cash position) constraint.
        """
        return w_plus >= 0.0


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
            The holding constraints based on the no short cash positions constraint. Assumes cash in last position.
        """
        return w_plus >= 0.0


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
            The holding constraints based on the equal long and short exposure constraint.
        """
        return sum(w_plus) == 0.0
