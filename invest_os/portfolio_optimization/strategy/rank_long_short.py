import pandas as pd
import datetime as dt

from invest_os.portfolio_optimization.strategy import BaseStrategy

class RankLongShort(BaseStrategy):
    def __init__(self):
        self.costs = []
        self.constraints = []

    def generate_trade_list(self, portfolio, t=dt.datetime.today()):
        pass