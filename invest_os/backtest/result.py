import pandas as pd
import numpy as np

class Result():
    def __init__(self):
        pass

    def save_data(self, name, t, entry):
        try:
            getattr(self, name).loc[t] = entry
        except AttributeError:
            setattr(self, name,
                    (pd.Series if np.isscalar(entry) else
                     pd.DataFrame)(index=[t], data=[entry]))

    def save_position(self, t, u, h_next):
        """
        u: trades for period
        h_next: holdings at beginning of period, after trades
        """
        self.save_data("u", t, u)
        self.save_data("h_next", t, h_next)

    @property
    def h(self):
        """
        Concatenate initial portfolio and h_next dataframe.
        """
        tmp = self.h_next.copy()
        tmp = self.h_next.shift(1) # Shift h_next to h timing
        return tmp[1:]

    @property
    def v(self):
        """The value of the portfolio over time.
        """
        return self.h.sum(axis=1)