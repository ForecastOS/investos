import pandas as pd
import numpy as np
import statistics
import datetime as dt

import investos.portfolio_optimization.strategy as strategy
import investos.backtest as backtest
import investos.util as util

class Optimizer():
    
    BASE_CONFIG = {
        "constraints": {
            "investing_style": {
                "long_only": True,
            },
            "risk": {
                "max_asset_weight": 1, # 1 == 100%
                "neutral_categories": [], # For neutralizing exposures to certain risk factors, like industries, small-cap, etc.
            },
            "borrowing": {
                "allowed": False,
                "leverage": 1, # i.e. absolute exposure
            },
            "trading": {
                "turnover_limit": None,
            },
        },
        "forecast": {
            "start": None, # Earliest date in df
            "stop": None, # Last date in df
            "std_dev": {
                "n_prev_periods": 100, # Calc from historical if not passed in
            },
            "spread": {
                "n_prev_periods": 100, # Calc from historical if not passed in
            },
            "volume": {
                "n_prev_periods": 100, # Calc from historical if not passed in
            },
        },
        "borrowing": {
            "interest_rate": 0.005,
        },
        "traiding": {
            "sensitivity": 1, 
            "asymmetry": 0, 
            "round_trip_costs_enabled": False, # Auto-enable for SPO; only for SPO
            "ignore": {
                "trading_costs": False,
                "holding_costs": False,
            },
        },
        "risk_model": {
            "aversion": 0.5,
            "covariance_risk_factors": 15, 
            "use_full_covariance_matrix": False,
        },
        "restricted": {
            "all": [],
            "short": [],
            "long": [],
        },
    }

    def __init__(
        self, 
        df_forecast,
        df_actual=None,
        df_historical=None, 
        strategy=strategy.RankLongShort,
        backtest_model=None,
        initial_portfolio=None, # In dollars (or other currency), not in weights
        costs=[],
        aum=100_000_000,
        df_categories=None, 
        config={},
        **kwargs):
        
        self.config = util.deep_dict_merge(self.BASE_CONFIG, config)

        self.strategy = strategy # Must be initialized first

        self.costs = costs
        
        if backtest_model is None:
            if df_actual is not None:
                self.backtest_model = backtest.Result
            else:
                self.backtest_model = backtest.ForecastResult
        else:   
            self.backtest_model = backtest_model # Not initialized when passed in

        self.backtest_model.optimizer = self

        if df_historical is not None:
            self.historical = {}
            self.historical['return'] = self.pivot_and_fill(df_historical, values='return')
            self.historical['price'] = self.pivot_and_fill(df_historical, values='price')
            self.historical['volume'] = self.pivot_and_fill(df_historical, values='volume')
            self.historical['spread'] = self.pivot_and_fill(df_historical, values='spread')

        self.forecast = {}
        self.forecast['return'] = self.pivot_and_fill(df_forecast, values='return')
        self.forecast['std_dev'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'std_dev'), values='std_dev')
        self.forecast['volume'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'volume'), values='volume')
        self.forecast['half_spread'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'spread'), values='spread') / 2
        self.forecast['price'] = self.create_price(self.forecast['return'])
        self._set_cash_return_for_each_period(self.forecast['return'])

        if df_actual is not None:
            self.actual = {}
            self.actual['return'] = self.pivot_and_fill(df_actual, values='return')
            self.actual['std_dev'] = self.pivot_and_fill(self.create_forecast(df_actual, 'std_dev'), values='std_dev')
            self.actual['volume'] = self.pivot_and_fill(self.create_forecast(df_actual, 'volume'), values='volume')
            self.actual['half_spread'] = self.pivot_and_fill(self.create_forecast(df_actual, 'spread'), values='spread') / 2
            self.actual['price'] = self.create_price(self.actual['return'])
            self._set_cash_return_for_each_period(self.actual['return'])

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
                self.historical['return'].tail(
                    self.config['forecast'][col_name]['n_prev_periods']
                ).std().rename(col_name).reset_index(), 
                how='left', 
                on='asset'
            )
        else:
            return df_forecast[['date', 'asset']].merge(
                self.historical[col_name].tail(
                    self.config['forecast'][col_name]['n_prev_periods']
                ).mean().rename(col_name).reset_index(), 
                how='left', 
                on='asset'
            )


    def create_price(self, df_return):
        last_historical_prices = self.historical['price'].loc[self.historical['price'].index.max()]
        
        # Make geometric - add 1
        df = df_return + 1
        
        # Geometric cumulative mean
        df = df.cumprod()
        
        # Multiply geometric return factors by last historical value
        return df * self.historical['price'].loc[self.historical['price'].index.max()]
       

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

        # Propograte through future trades and resulting positions
        for t in self.forecast['return'].index:
            u = self.strategy.generate_trade_list(h_next, t)
            h_next, u = self.propagate(h_next, u, t)
            self.backtest.save_position(t, u, h_next)

        print("Done simulating.")
        
        return self.backtest


    def propagate(self, h, u, t):
        """From CvxPortfolio

        Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: portfolio after returns propagation (for t to t+1 period)
            u: trades vector with simulated cash balance
        """
        h_plus = h + u
        
        costs = [cost.value_expr(t, h_plus=h_plus, u=u) for cost in self.strategy.costs]

        u["cash"] = -sum(u[u.index != "cash"]) - sum(costs)
        h_plus["cash"] = h["cash"] + u["cash"]

        h_next = self.actual['return'].loc[t] * h_plus + h_plus

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
        
        for c in self.strategy.costs:
            c.optimizer = self

# Unwind trade from LongShort appropriately. Probably with cumprod (see create_forecast_price method)

# -->--> H cost
# -->--> T cost

# [ ] Build crude inv.backtest.Result model to make sure everything is working

# STOP THURSDAY

# [ ] Duck type everything and allow everything to be passed in (for easy extensibility)
# --> Duck typing initial transform for historical and forecast data would be amazing as well (so it can be customized and resused by specific companies)

# [ ] Should only NEED forecast, not historical, if everything is passed in

# [ ] Build duck-typed (i.e. swappable) risk model

# [ ] Build duck-typed (i.e. swappable) cost model
# --> for t costs
# --> for h costs

# [ ] Run SPO - as separate class

# [ ] Run MPO - as separate class

# [ ] Add ability to run multiple iterations of type, config, etc.
# --> Essentially grid search wrapper

# [ ] Support forecast dividends / distributions (positive or negative)

# NOTE TO SELF: backtester can AND SHOULD sometimes have different costs than strategy. Strategy costs are for informing trades (i.e. discouraging turnover), backtest costs are ACTUAL expected costs given trades determined by strategy