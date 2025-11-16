# http://web.cvxr.com/cvx/doc/basics.html#constraints
from investos.util import remove_excluded_columns_np


class BaseConstraint:
    """
    Base class for constraint objects used in convex portfolio optimization strategies.

    Subclass `BaseConstraint`, and create your own `cvxpy_expression` method to create custom constraints.
    """

    def __init__(self, **kwargs):
        # Can only have exclude or include.
        # Not sensible to have both.
        self.exclude_assets = kwargs.get("exclude_assets", [])
        self.include_assets = kwargs.get("include_assets", [])

    def cvxpy_expression(
        self,
        t,
        weights_portfolio_plus_trades,
        weights_trades,
        portfolio_value,
        asset_idx,
    ):
        weights_portfolio_plus_trades = remove_excluded_columns_np(
            weights_portfolio_plus_trades,
            asset_idx,
            include_assets=self.include_assets,
            exclude_assets=self.exclude_assets,
        )
        weights_trades = remove_excluded_columns_np(
            weights_trades,
            asset_idx,
            include_assets=self.include_assets,
            exclude_assets=self.exclude_assets,
        )

        return self._cvxpy_expression(
            t, weights_portfolio_plus_trades, weights_trades, portfolio_value
        )

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        raise NotImplementedError

    def metadata_dict(self):
        metadata_dict = {}

        if getattr(self, "limit", None):
            metadata_dict["limit"] = self.limit

        return metadata_dict
