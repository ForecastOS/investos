import pandas as pd
import numpy as np

import datetime as dt
import collections

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
    def summary(self):
        print(self._summary_string())

    def _summary_string(self):
        data = collections.OrderedDict({
            'Number of periods':
                self.u.shape[0],
            'Initial timestamp':
                self.h.index[0],
            'Final timestamp':
                self.h.index[-1],
            'Annualized portfolio return (%)':
                str(round(self.annualized_return * 100, 2)) + '%',
            # 'Excess return (%)':
            #     self.excess_returns.mean() * 100 * self.PPY,
            # 'Excess risk (%)':
            #     self.excess_returns.std() * 100 * np.sqrt(self.PPY),
            'Sharpe ratio':
                self.sharpe_ratio,
            'Max drawdown':
                f"{round(self.max_drawdown, 2)}%",
            'Annual turnover (x)':
                str(round(self.turnover.mean() * self.ppy, 2)) + 'x',
        })

        return (pd.Series(data=data).
                to_string(float_format='{:,.3f}'.format))


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


    @property
    def returns(self):
        """The returns R_t = (v_{t+1}-v_t)/v_t
        """
        val = self.v
        return pd.Series(data=val.values[1:] / val.values[:-1] - 1,
                         index=val.index[1:]).dropna()


    @property
    def annualized_return(self):
        a_return = self.v[-1] / self.v[0]

        return (
            ( a_return ** (1 / self.years_forecast) ) - 1
        )


    @property
    def excess_returns(self):
        return (self.returns - self.optimizer.actual['return']['cash']).dropna()


    @property
    def years_forecast(self):
        return (self.v.index[-1] - self.v.index[0]) / dt.timedelta(365,0,0,0)


    @property
    def ppy(self):
        return self.v.shape[0] / self.years_forecast


    @property
    def sharpe_ratio(self):
        return (
            np.sqrt(self.ppy) * np.mean(self.excess_returns) /
            np.std(self.excess_returns)
        )


    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t
        """
        noncash_trades = self.u.drop(['cash'], axis=1)
        return np.abs(noncash_trades).sum(axis=1) / self.v


    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent.
        """
        val_arr = self.v.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif 100 * (cur_max - val) / cur_max > max_dd_so_far:
                max_dd_so_far = 100 * (cur_max - val) / cur_max
        return max_dd_so_far