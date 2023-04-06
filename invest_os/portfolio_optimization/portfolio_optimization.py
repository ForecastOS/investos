import pandas as pd

import invest_os as inv
import invest_os.util as util

class PortfolioOptimization():
    
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
            "interest_rate_on_cash": None,
            "interest_rate_on_assets": None,
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
        df_historical, 
        df_forecast,
        strategy=inv.portfolio_optimization.strategy.RankLongShort,
        initial_portfolio=None, # In dollars (or other currency), not in weights
        initial_cash=100_000_000,
        df_categories=None, 
        config={},
        **kwargs):
        self.strategy = strategy

        self.config = util.deep_dict_merge(self.BASE_CONFIG, config)

        self.historical = {}
        self.historical['return'] = self.pivot_and_fill(df_historical, values='return')
        self.historical['price'] = self.pivot_and_fill(df_historical, values='price')
        self.historical['volume'] = self.pivot_and_fill(df_historical, values='volume')
        self.historical['spread'] = self.pivot_and_fill(df_historical, values='spread')

        self.forecast = {}
        self.forecast['return'] = self.pivot_and_fill(df_forecast, values='return')
        self.forecast['std_dev'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'std_dev'), values='std_dev')
        self.forecast['volume'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'volume'), values='volume')
        self.forecast['spread'] = self.pivot_and_fill(self.create_forecast(df_forecast, 'spread'), values='spread')
        self.forecast['price'] = self.create_forecast_price()


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


    def create_forecast_price(self):
        last_historical_prices = self.historical['price'].loc[self.historical['price'].index.max()]
        
        # Make geometric - add 1, 
        df = self.forecast['return'] + 1
        
        # Geometric cumulative mean
        df = df.cumprod()
        
        # Multiply geometric return factors by last historical value
        return df * self.historical['price'].loc[self.historical['price'].index.max()]
       

    def optimize(self):
        print("Optimizing...")

        for t in self.forecast['return'].index:
            self.strategy.generate_trade_list(self, t)

        print("Done optimizing. Check backtest_result object for more information")

# START THURSDAY

# [ ] Default initial portfolio to 0, with X in cash
# --> [ ] Have default value for cash too (100MM) that can be overridden
# --> [ ] Add cash myself to initial portfolio DF 
# --> [ ] Add default borrow cost annually


# [ ] RankLongShort
# --> Generate trade list
# --> propogate (which will calc holding and trade costs)
# -->--> New position and trades
# -->--> H cost
# -->--> T cost
# --> Log result (in result - backtest - object). Borrow heavy inspo to keep easy

# [ ] Build crude inv.backtest.Result model to make sure everything is working
# --> inspo from result

# STOP THURSDAY

# [ ] Need to water down signal more

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