import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint
from investos.util import values_in_time


class ZeroFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, **kwargs):
        self.factor_exposure = factor_exposure
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), w_plus)) == 0
        )


class ZeroTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, **kwargs):
        self.factor_exposure = factor_exposure
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), z)) == 0


class MaxFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), w_plus))
            <= self.limit
        )


class MinFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=-0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), w_plus))
            >= self.limit
        )


class MaxAbsoluteFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.4, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(
                cvx.multiply(values_in_time(self.factor_exposure, t), cvx.abs(w_plus))
            )
            <= self.limit
        )


class MaxAbsoluteTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.01, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), cvx.abs(z)))
            <= self.limit
        )


class MaxTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), z))
            <= self.limit
        )


class MinTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=-0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        return (
            cvx.sum(cvx.multiply(values_in_time(self.factor_exposure, t), z))
            >= self.limit
        )
