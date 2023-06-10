import pandas as pd
import numpy as np
import statistics
import datetime as dt

import investos.portfolio_optimization.strategy as strategy
from investos.portfolio_optimization.strategy import BaseStrategy, RankLongShort
import investos.backtest as backtest
import investos.util as util

class Backtester():
    """Container class that runs backtests using passed-in portfolio optimization `strategy` (see :py:class:`~investos.portfolio_optimization.strategy.base_strategy.BaseStrategy`), then saves results into passed-in `result` (see :py:class:`~investos.backtest.result.Result`) class.
    
    Parameters
    ----------
    df_forecast : pd.DataFrame
        DataFrame of forecast asset returns for backtest period.

        Expected columns: 
        asset (unique ID/ticker), 
        date (datetime), 
        return (float)
    
    df_actual : pd.DataFrame
        DataFrame of actual asset returns for backtest period. Used to create forecasts for std_dev, spread, and volume if forecasts not given.

        Expected columns: 
        asset (unique ID/ticker), 
        date (datetime), 
        price (float), 
        return (float), 
        volume (of shares traded, float), 
        spread (float)

    strategy : :py:class:`~investos.portfolio_optimization.strategy.base_strategy.BaseStrategy`
        Optimization strategy used by backtester. Used to determine what trades to make at each forecast time period.
    
    backtest_model : :py:class:`~investos.backtest.result.Result`
        Stores result from simulated backtest, and contains convenience properties and methods for reporting.

    initial_portfolio : pd.DataFrame, optional
        Initial portfolio values in dollars (or other currency), not in weights.

    aum : float, optional
        Total assets under management, in dollars (or other currency).

    df_categories : pd.DataFrame, optional
        Additional category data for assets. Used by certain optimization strategies.

    config : dict, optional
        Configuration parameters.

    **kwargs : 
        Additional keyword arguments.

    Attributes
    ----------
    config : dict
        Configuration parameters after merging with base configuration.

    strategy : :py:class:`~investos.portfolio_optimization.strategy.base_strategy.BaseStrategy`
        Optimization strategy used by backtester.

    backtest_model : :py:class:`~investos.backtest.result.Result`
        Stores result from simulated backtest.

    forecast : dict
        Holds the pivoted and filled forecast DataFrames.

    actual : dict
        Holds the pivoted and filled actual DataFrames, if provided.

    initial_portfolio : pd.DataFrame
        Initial portfolio values, including cash.

    Methods
    -------
    optimize(self):
        Optimizes the portfolio and backtests it. Returns :py:class:`~investos.backtest.result.Result` object.
    
    pivot_and_fill(self, df: pd.DataFrame, values: str, columns='asset', index='date', fill_method='bfill'):
        Pivots and fills the DataFrame based on the provided values, columns, and index.

    create_forecast(self, df_forecast: pd.DataFrame, col_name: str = 'std_dev'):
        Creates a forecast DataFrame based on the provided df_forecast and col_name.

    create_price(self, df_return: pd.DataFrame):
        Creates price data based on provided return data and last historical price.

    create_initial_portfolio(self, initial_portfolio, aum):
        Creates the initial portfolio based on provided initial portfolio (or aum value if iniitial portfolio not provided).

    propagate(self, h: pd.Series, u: pd.Series, t: dt.Series):
        Heavily inspired by CvxPortfolio.

        Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n pandas Series vector with stock trades
            t: current datetime

        Returns:
            h_next: pandas Series portfolio after returns propagation (for t to t+1 period)
            u: pandas Series trades vector with simulated cash balance

    get_initial_t(self):
        Gets the initial time period for the backtest.
    """
    
    BASE_CONFIG = {
        "forecast": {
            "std_dev": {
                "calc_from_n_prev_periods": 100, # Calc from actual_df if not passed in
            },
            "half_spread": {
                "calc_from_n_prev_periods": 100, # Calc from actual_df if not passed in
            },
            "volume": {
                "calc_from_n_prev_periods": 100, # Calc from actual_df if not passed in
            },
        },
        "borrowing": {
            "interest_rate": 0.005,
            "short_rate": 0.005
        },
    }

    def __init__(
        self, 
        df_forecast: pd.DataFrame,
        df_actual: pd.DataFrame = None,
        strategy: BaseStrategy = RankLongShort,
        backtest_model: backtest.Result = None,
        initial_portfolio: pd.DataFrame = None, # In dollars (or other currency), not in weights
        aum: float = 100_000_000,
        df_categories: pd.DataFrame = None, 
        start_date = None,
        end_date = None,
        config: dict = {},
        **kwargs):
        
        self.config = util.deep_dict_merge(self.BASE_CONFIG, config)

        self.strategy = strategy # Must be initialized first
        if getattr(self.strategy, 'risk_model', None):
            self.strategy.costs += [self.strategy.risk_model]

        if backtest_model is None:
            if df_actual is not None:
                self.backtest_model = backtest.Result
            else:
                self.backtest_model = backtest.ForecastResult
        else:   
            self.backtest_model = backtest_model # Not initialized when passed in

        self.backtest_model.optimizer = self

        self.forecast = {}
        self.forecast['date'] = {
            "start": start_date or df_forecast.date.min(),
            "end": end_date or df_forecast.date.max(),
        }
        df_forecast = df_forecast[
            (df_forecast['date'] >= self.forecast['date']['start']) & 
            (df_forecast['date'] <= self.forecast['date']['end'])
        ]
        
        if df_actual is not None:
            self.actual = {}
            self.actual['return'] = self.pivot_and_fill(df_actual, values='return')
            self.actual['price'] = self.pivot_and_fill(df_actual, values='price')
            self.actual['volume'] = self.pivot_and_fill(df_actual, values='volume')
            self.actual['half_spread'] = (
                self.pivot_and_fill(df_actual, values='spread') / 2
            ).rename(
                columns={'spread': 'half_spread'}
            )
            self.actual['std_dev'] = self.pivot_and_fill(self.create_forecast(df_actual, 'std_dev'), values='std_dev')
            self._set_cash_return_for_each_period(self.actual['return'])

        self.forecast['return'] = self.pivot_and_fill(df_forecast, values='return')
        self.forecast['std_dev'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'std_dev'), values='std_dev')
        self.forecast['volume'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'volume'), values='volume')
        self.forecast['half_spread'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'half_spread'), values='half_spread')
        self.forecast['price'] = self.create_price(self.forecast['return'])
        self._set_cash_return_for_each_period(self.forecast['return'])

        self.create_initial_portfolio(initial_portfolio, aum)

        self._set_references_back_to_optimizer()


    def pivot_and_fill(self, df, values, columns='asset', index='date', fill_method='bfill'):
        return pd.pivot(
            df, values=values, columns=columns, index=index
        ).fillna(method=fill_method)


    def create_forecast(self, df_forecast, col_name='std_dev'):
        if col_name in df_forecast.columns:
            return df_forecast
        elif col_name == 'std_dev':
            return df_forecast[['date', 'asset']].merge(
                self.actual['return'][self.actual['return'].index < self.forecast['date']['start']].tail(
                    self.config['forecast'][col_name]['calc_from_n_prev_periods']
                ).std().rename(col_name).reset_index(), 
                how='left', 
                on='asset'
            )
        else:
            return df_forecast[['date', 'asset']].merge(
                 self.actual[col_name][self.actual[col_name].index < self.forecast['date']['start']].tail(
                    self.config['forecast'][col_name]['calc_from_n_prev_periods']
                ).mean().rename(col_name).reset_index(), 
                how='left', 
                on='asset'
            )


    def create_price(self, df_return):
        last_historical_prices = (
            self.actual['price']
                .loc[self.actual['price'][self.actual['price'].index < self.forecast['date']['start']].index.max()]
            )
        
        # Make geometric - add 1
        df = df_return + 1
        
        # Geometric cumulative mean
        df = df.cumprod()
        
        # Multiply geometric return factors by last historical value
        return df * last_historical_prices
       

    def create_initial_portfolio(self, initial_portfolio, aum):
        if initial_portfolio is None:
            initial_portfolio = pd.Series(index=self.forecast['return'].columns, data=0)
            initial_portfolio["cash"] = aum

        self.initial_portfolio = initial_portfolio


    def optimize(self):
        print("Optimizing...")

        self.backtest = self.backtest_model()

        # Submit initial position
        t = self.get_initial_t()
        u = pd.Series(index=self.initial_portfolio.index, data=0) # No trades at time 0
        h_next = self.initial_portfolio # Includes cash
        self.backtest.save_position(t, u, h_next)

        # Propagate through future trades and resulting positions
        for t in self.forecast['return'].index:
            u = self.strategy.generate_trade_list(h_next, t)
            h_next, u = self.propagate(h_next, u, t)
            self.backtest.save_position(t, u, h_next)

        print("Done simulating.")
        
        return self.backtest


    def propagate(self, h, u, t):
        # print(f"Propagating for {t}")
        
        h_plus = h + u
        
        costs = [cost.value_expr(t, h_plus=h_plus, u=u) for cost in self.strategy.costs]

        u["cash"] = -sum(u[u.index != "cash"]) - sum(costs)
        h_plus["cash"] = h["cash"] + u["cash"]

        h_next = self.actual['return'].loc[t] * h_plus + h_plus

        # TBU - is this needed?
        zero_threshold = 0.00001
        h_next[np.abs(h_next) < zero_threshold] = 0
        u[np.abs(u) < zero_threshold] = 0
        
        return h_next, u


    def get_initial_t(self):
        median_time_delta = statistics.median(
            self.forecast['return'].index[1:5] - self.forecast['return'].index[0:4]
        )

        return self.forecast['return'].index[0] - median_time_delta


    def _set_cash_return_for_each_period(self, df): 
        df['cash'] = self.config['borrowing']['interest_rate']

        df['tmp_date'] = df.index
        df['tmp_date_lagged'] = df['tmp_date'].shift(1)
        df['tmp_date_delta'] = df['tmp_date'] - df['tmp_date_lagged']
        df['tmp_date_delta_fraction_of_year'] = df['tmp_date_delta'] / dt.timedelta(365,0,0,0)
        df['cash'] = (1 + df['cash']) ** df['tmp_date_delta_fraction_of_year'] - 1
        df['cash'] = df['cash'].fillna(method='bfill')

        df.drop(columns=['tmp_date', 'tmp_date_lagged', 'tmp_date_delta', 'tmp_date_delta_fraction_of_year'], inplace=True)


    def _set_references_back_to_optimizer(self):
        self.strategy.forecast_returns = self.forecast['return']
        self.strategy.optimizer = self
        
        for c in self.strategy.costs + (self.strategy.constraints or []):
            c.optimizer = self
