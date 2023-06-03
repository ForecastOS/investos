from investos.util import values_in_time

import cvxpy as cvx
import numpy as np

# http://web.cvxr.com/cvx/doc/basics.html#constraints
class BaseConstraint(object):
    """
    Base class for constraint objects used in convex portfolio optimization strategies.

    Subclass `BaseConstraint`, and create your own `weight_expr` method to create custom constraints.
    """
    def __init__(self, **kwargs):
        self.optimizer = None # Set during Optimizer initialization


    def weight_expr(self, t, w_plus, z, v):
        raise NotImplementedError