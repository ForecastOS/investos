import pandas as pd
import numpy as np

from investos.backtest import Result

class ForecastResult(Result):
    """To be implemented.

    For forecast results (where future prices are unknown). Used to generate foward-looking trades / ideal positions.
    """
    def __init__(self):
        pass

# --> Sometimes (during backtest) you know actual (market) prices before
# --> Sometimes (during construction for future that hasn't happened yet) you don't
# --> SO backtest.result OBJECT NEEDS ACTUAL (MARKET) PRICES AS WELL, whereas...
# --> ... backtest.forecast_result does not
# --> backtest.forecast_result should have backtest.result as parent, but remind user that results aren't real because actual prices not provided