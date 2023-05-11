class BaseCost():
    def __init__(self):
        pass

    def value_expr(self, t, h_plus, u):
        raise NotImplementedError