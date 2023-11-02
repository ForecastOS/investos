<h1>Analyzing Backtest Results</h1>

The [BaseResult](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/result/base_result.py) class captures portfolio data and calculates performance metrics for your investment strategy.

An instance of BaseResult is returned by calling the method `generate_positions()` on an instance of BacktestController.

## Retrieving a Portfolio Summary

To get a string summary of your backtest result:

```python
result_instance.summary
```

## Reporting Properties

BaseResult instances have the following properties for performance reporting:

1. `actual_returns`: Dataframe of actual returns for assets.
2. `annualized_benchmark_return`: Annualized return for the benchmark over the entire period.
3. `annualized_excess_return`: Annualized excess return for the entire period.
4. `annualized_return`: Annualized return for the entire period under review.
5. `annualized_return_over_cash`: Annualized return over cash for the entire period.
6. `benchmark_returns`: Series of returns for the benchmark.
7. `benchmark_v`: Series of simulated portfolio values if invested 100% in the benchmark at time 0.
8. `cash_column_name`: String of cash column name in holdings and trades.
9. `excess_returns`: Series of returns in excess of the benchmark.
10. `excess_risk_annualized`: Risk in excess of the benchmark.
11. `h`: Dataframe of asset holdings at the beginning of each datetime period.
12. `information_ratio`: (Annualized) Information Ratio of the portfolio.
13. `num_periods`: Number of periods in the backtest.
14. `portfolio_hit_rate`: Proportion of periods in which the portfolio had positive returns.
15. `ppy`: Float representing the number of periods per year in the backtest period.
16. `returns`: Series of the returns for each datetime period compared to the previous period.
17. `returns_over_cash`: Series of returns in excess of risk-free returns.
18. `risk_free_returns`: Series of risk-free returns.
19. `risk_over_cash_annualized`: Risk in excess of the risk-free rate.
20. `sharpe_ratio`: (Annualized) Sharpe Ratio of the portfolio.
21. `total_benchmark_return`: Return over the benchmark for the entire period under review.
22. `total_excess_return`: Excess return for the entire period (portfolio return minus benchmark return).
23. `total_return`: Total return for the entire period under review.
24. `total_return_over_cash`: Total returns over cash for the entire period under review.
25. `total_risk_free_return`: Total return over the risk-free rate for the entire period.
26. `trades`: Series of trades (also available as `u`).
27. `v`: Series of the value of the portfolio for each datetime period.
28. `v_with_benchmark`: Dataframe with simulated portfolio and benchmark values.
29. `years_forecast`: Float representing the number of years in the backtest period.

## Extend BaseResult For Custom Reporting

If the BaseResult class is missing a property or method you wish it had, you can easily extend the class (i.e. create a new class that inherits from BaseResult) and add your desired functionality!

BacktestController accepts a `results_model` kwarg (note: if passed, it expects the model to be instantiated). Don't be afraid to roll out your own custom reporting functionality!

## Next: Analyzing External Portfolios

It's possible to backtest a portfolio created outside of InvestOS. Let's review how that works: [Analyzing External Portfolios](/guides/reporting/external_portfolios).
