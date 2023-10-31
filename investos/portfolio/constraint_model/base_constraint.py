# http://web.cvxr.com/cvx/doc/basics.html#constraints
from investos.util import remove_excluded_columns_np


class BaseConstraint:
    """
    Base class for constraint objects used in convex portfolio optimization strategies.

    Subclass `BaseConstraint`, and create your own `weight_expr` method to create custom constraints.
    """

    def __init__(self, **kwargs):
        # Can only have exclude or include.
        # Not sensible to have both.
        self.exclude_assets = kwargs.get("exclude_assets", [])
        self.include_assets = kwargs.get("include_assets", [])

    def weight_expr(self, t, w_plus, z, v, asset_idx):
        w_plus = remove_excluded_columns_np(
            w_plus,
            asset_idx,
            include_assets=self.include_assets,
            exclude_assets=self.exclude_assets,
        )
        z = remove_excluded_columns_np(
            z,
            asset_idx,
            include_assets=self.include_assets,
            exclude_assets=self.exclude_assets,
        )

        return self._weight_expr(t, w_plus, z, v)

    def _weight_expr(self, t, w_plus, z, v):
        raise NotImplementedError

    def metadata_dict(self):
        metadata_dict = {}

        if getattr(self, "limit", None):
            metadata_dict["limit"] = self.limit

        return metadata_dict
