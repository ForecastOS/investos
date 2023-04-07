import pandas as pd
import numpy as np

from invest_os.backtest import Result

class ForecastResult(Result):
    def __init__(self):
        pass

# --> Sometimes (during backtest) you know actual (market) prices before
# --> Sometimes (during construction for future that hasn't happened yet) you don't
# --> SO backtest.result OBJECT NEEDS ACTUAL (MARKET) PRICES AS WELL, whereas...
# --> ... backtest.forecast_result does not
# --> backtest.forecast_result should have backtest.result as parent, but remind user that results aren't real because actual prices not provided