import pandas as pd

class BaseStrategy():
    def __init__(self):
        self.costs = []
        self.constraints = []

    def generate_trade_list(self):
        raise NotImplementedError

    def generate_trade_list_weights(self):
        pass

    def generate_trade_list_shares(self):
        pass