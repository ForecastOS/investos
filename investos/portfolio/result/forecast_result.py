import pandas as pd
import numpy as np

from investos.backtest import Result

class ForecastResult(Result):
    """To be implemented.

    For forecast results (where future prices are unknown). Used to generate foward-looking trades / ideal positions.
    """
    def __init__(self):
        pass