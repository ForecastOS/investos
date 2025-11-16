<h1>Using the BacktestController Class</h1>

The [BacktestController](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/backtest_controller.py) class is responsible for running backtests on your chosen investment strategy and saving the results.

A step-by-step guide follows for instantiating this class:

## Steps to Instantiate

### Prepare your Investment Strategy:

Before instantiating `BacktestController`, you should already have an instance of your desired investment strategy:

```python
import investos as inv

strategy = inv.portfolio.strategy.SPO(
    actual_returns = actual_returns,
    forecast_returns = forecast_returns,
    costs = [
        ShortHoldingCost(
            short_rates=short_rates,
            exclude_assets=["cash"]
        )
    ],
    constraints = [
        MaxShortLeverageConstraint(limit=0.3),
        MaxLongLeverageConstraint(limit=1.3),
        MinWeightConstraint(),
        MaxWeightConstraint(),
        LongCashConstraint()
    ],
    cash_column_name="cash"
)
```

### Instantiate the `BacktestController` Class:

Use the strategy instance from the previous step as the first argument.

```python
backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```

### Provide Optional Arguments (if necessary):

-   `hooks`: A dictionary of hooks for specific events during the backtest. Currently, the only hook that exists is `after_trades`. You can use it to do things like increase / decrease leverage based on performance, change constraints on the fly, print output on each iteration, etc.

```python
hooks = {
    "after_trades": [
        lambda backtest, t, dollars_trades, dollars_holdings_at_next_t: print(".", end=''),
    ]
}
```

-   `initial_portfolio`: A `pandas.Series` indicating the initial portfolio allocation. If not provided, the controller will create a default initial portfolio with an AUM of 100,000,000 (allocated to cash) and no other initial allocations.

-   `results_model`: A custom result class to store backtest results. It defaults to `BaseResult` if not provided.

-   `time_periods`: A series of time periods you wish to backtest. By default, it uses `actual_returns.index` from your investment strategy instance.

-   `start_date` and `end_date`: Start and end dates for the backtest. It uses the first and last dates from `time_periods` if not set.

-   `aum`: The initial portfolio's asset under management. Defaults to $100MM.

Example instantiation with optional arguments:

```python
backtest = BacktestController(
    strategy=strategy,
    start_date='2020-01-01',
    end_date='2021-01-01',
    aum=50_000_000,
    hooks={
        'after_trades': [custom_hook_function]
    },
)
```

### Run the Backtest:

Once you've instantiated `BacktestController`, you can run the backtest using `generate_positions`.

This returns an instance of BaseResult for performance reporting.

```python
results = backtest.generate_positions()
# result.summary will print high-level results
```

## Conclusion

With the steps mentioned above, you can effectively instantiate the `BacktestController` class and execute a backtest with your desired strategy.

Adjust the optional parameters according to your needs to fine-tune your backtesting process!

## Next: Investment Strategies

Next, let's explore the off-the-shelf investment strategies available to you: [Off-The-Shelf Investment Strategies](/guides/off_the_shelf/investment_strategies).
