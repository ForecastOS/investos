import statistics
import time
from datetime import datetime

import pandas as pd
from dask.distributed import Client
from dask_cloudprovider.aws import FargateCluster

from investos.portfolio.result import BaseResult


class BacktestController:
    """Container class that runs backtests using passed-in portfolio engineering `strategy` (see :py:class:`~investos.portfolio.strategy.base_strategy.BaseStrategy`), then saves results into `result` (see :py:class:`~investos.backtest.result.Result`) class."""

    def __init__(self, strategy, **kwargs):
        self.strategy = strategy
        self.strategy.backtest_controller = self

        # Optional
        self._set_time_periods(**kwargs)

        self.hooks = kwargs.get("hooks", {})

        self.distributed = kwargs.get("distributed", False)
        self.dask_cluster = kwargs.get("dask_cluster", False)
        self.dask_cluster_config = {
            "n_workers": 50,
            "image": "daskdev/dask:latest",
            "region_name": "us-east-2",  # Change this to your preferred AWS region
            "worker_cpu": 1024 * 2,  # 2 vCPU
            "worker_mem": 1024 * 4,  # 4 GB memory
            "scheduler_cpu": 1024 * 16,
            "scheduler_mem": 1024 * 32,
            "scheduler_timeout": "3600s",
            "environment": {
                "EXTRA_PIP_PACKAGES": "investos scikit-learn",
            },
        }
        self.dask_cluster_config.update(kwargs.get("dask_cluster_config", {}))

        self.initial_portfolio = kwargs.get(
            "initial_portfolio",
            self._create_initial_portfolio_if_not_provided(**kwargs),
        )

        # Create results instance for saving performance
        self.results = kwargs.get("results_model", BaseResult)(
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.results.strategy = self.strategy

    def generate_positions(self):
        print(
            f"Generating historical portfolio trades and positions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}..."
        )
        # Create t == 0 position (no trades)
        t = self._get_initial_t()
        u = pd.Series(index=self.initial_portfolio.index, data=0)
        h_next = self.initial_portfolio  # Includes cash
        self.results.save_position(t, u, h_next)

        if self.distributed:
            self._dask_start_client_and_cluster()
            self.strategy.precompute_trades_distributed(h_next, self.time_periods)
            print("\nClosing dask cluster...")
            self.dask_cluster.close()
            print("\nDask cluster closed.\n")

        # Walk through time and calculate future trades, estimated and actual costs and returns, and resulting positions
        for t in self.time_periods:
            u = self.strategy.generate_trade_list(h_next, t)
            h_next, u = self.strategy.get_actual_positions_for_t(h_next, u, t)
            self.results.save_position(t, u, h_next)

            for func in self.hooks.get("after_trades", []):
                func(self, t, u, h_next)

        print(f"\n\nDone simulating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        return self.results

    def _set_time_periods(self, **kwargs):
        time_periods = kwargs.get("time_periods", self.strategy.actual_returns.index)
        self.start_date = kwargs.get("start_date", time_periods[0])
        self.end_date = kwargs.get("end_date", time_periods[-1])

        self.time_periods = time_periods[
            (time_periods >= self.start_date) & (time_periods <= self.end_date)
        ]

    def _create_initial_portfolio_if_not_provided(self, **kwargs):
        aum = kwargs.get("aum", 100_000_000)
        initial_portfolio = pd.Series(
            index=self.strategy.actual_returns.columns, data=0
        )
        initial_portfolio[self.strategy.cash_column_name] = aum

        return initial_portfolio

    def _get_initial_t(self):
        try:
            median_time_delta = statistics.median(
                self.time_periods[1:5] - self.time_periods[0:4]
            )
        except ValueError:
            median_time_delta = self.time_periods[1] - self.time_periods[0]

        return pd.to_datetime(self.start_date) - median_time_delta

    def _dask_start_client_and_cluster(self, retries=5, delay=15):
        print(
            "\nDistributing trade generation with Dask client.\n\nTrade-specific costs, constraints, and risk-models will continue to work as expected."
        )
        if self.dask_cluster:
            self.client = Client(self.dask_cluster, timeout="3600s")
        else:
            for i in range(retries):
                try:
                    print(
                        f"\nCreating Dask cluster at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}..."
                    )
                    self.dask_cluster = FargateCluster(**self.dask_cluster_config)
                    self.client = Client(self.dask_cluster, timeout="600s")
                    print(
                        f"\nCluster created. Distributing tasks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                    )
                    return True
                except Exception as e:
                    print(
                        f"Connection attempt {i+1} failed: {e}. Will retry in {delay}s"
                    )
                    if i < retries - 1:
                        time.sleep(delay)
                    else:
                        raise
