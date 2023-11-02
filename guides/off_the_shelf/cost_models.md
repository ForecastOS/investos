<h1>Using Off-The-Shelf Cost Models</h1>

## Usage

Using cost models is simple!

You simply pass instantiated cost model instances into your desired investment strategy:

```
from investos.portfolio.strategy import YourDesiredStrategy

strategy = YourDesiredStrategy(
    actual_returns = actual_returns,
    ...,
    costs=[
        CostModelA(*args, **kwargs),
        CostModelB(*args, **kwargs)
    ],
)
```

and that's it!

## Optional Arguments

All cost models take the following optional arguments:

-   exclude_assets: [str]
-   include_assets: [str]
    -   Can't be used with exclude_assets
-   gamma: float = 1
    -   Gamma doesn't impact actual costs
    -   Gamma only (linearly) increases estimated costs during convex optimization trade list generation
    -   If you aren't using a convex optimization investment strategy, gamma does nothing

---

InvestOS provides the following cost models:

## ShortHoldingCost

[ShortHoldingCost](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/cost_model/short_holding_cost.py) calculates per period cost for holding short positions, given customizable short_rate.

To instantiate ShortHoldingCost you will need to set the following arguments:

-   short_rates: pd.DataFrame | pd.Series | float,

## TradingCost

[TradingCost](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/cost_model/trading_cost.py) calculates per period cost for trades based on forecast spreads, standard deviations, volumes, and actual prices.

To instantiate TradingCost you will need to set the following arguments:

-   forecast_volume: pd.DataFrame | pd.Series,
-   forecast_std_dev: pd.DataFrame | pd.Series,
-   actual_prices: pd.DataFrame,
-   sensitivity_coeff: float = 1
    -   For scaling transaction cost from market impact
    -   1 assumes trading 1 day's volume moves asset price by 1 forecast standard deviation in returns
-   half_spread: pd.DataFrame | pd.Series | float
    -   Half of forecast spread between bid and ask for each asset
    -   This model assumes half_spread represents the cost of executing a trade

## Next: The Choice Is Yours

Want to explore creating your own custom cost model? Check out [Custom Cost Models](/guides/bespoke/custom_cost_models).

Want to learn more about using constraint models? Check out [Constraint Models](/guides/off_the_shelf/constraint_models).
