import requests

import investos


class SaveResult:
    def save(
        self,
        description,
        tags=[],
        team_ids=[],
        strategy=None,
        forecast_ids=[],
    ):
        self.api_key = investos.api_key
        self.api_endpoint = investos.api_endpoint
        self.request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.tags = tags
        self.team_ids = team_ids

        self.save_backtest(description, strategy, forecast_ids)
        self.save_backtest_charts()

    def save_backtest(self, description, strategy=None, forecast_ids=[]):
        strategy = getattr(self, "strategy", strategy)

        json_body = {
            "backtest": {
                "description": description,
                "performance_summary": {
                    "total_return": self.total_return,
                    "annualized_return": self.annualized_return,
                    "sharpe_ratio": self.sharpe_ratio,
                    "information_ratio": self.information_ratio,
                    "max_drawdown": self.max_drawdown,
                    "annual_turnover": self.annual_turnover,
                    "portfolio_hit_rate": self.portfolio_hit_rate,
                    "annualized_excess_risk": self.excess_risk_annualized,
                },
                "tags": self.tags,
                "team_ids": self.team_ids,
                "portfolio_construction": strategy and strategy.metadata_dict(),
                "forecast_ids": forecast_ids,
            }
        }

        response = requests.post(
            f"{self.api_endpoint}/backtests",
            headers=self.request_headers,
            json=json_body,
        )

        if (
            response.status_code // 100 == 2
        ):  # Check if the status code is in the 200 range
            self.backtest_id = response.json().get("id")
            print(f"Backtest {self.backtest_id} saved. Creating charts.")
        else:
            print(
                f"Backtest API request failed with status code: {response.status_code}"
            )

    def save_backtest_charts(self):
        self.save_chart_historical_value()
        self.save_cumulative_returns()
        self.save_chart_rolling_sharpe()
        self.save_chart_historical_leverage()

    def save_chart_historical_value(self):
        json_body = {
            "chart": {
                "title": "Value evolution",
                "chartable_type": "Backtest",
                "chartable_id": self.backtest_id,
                "chart_traces": [
                    {
                        "x_name": "Dates",
                        "y_name": "Portfolio",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(self.v.values),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": "Benchmark",
                        "x_values": [str(el) for el in self.benchmark_v.index],
                        "y_values": list(self.benchmark_v.values),
                    },
                ],
            }
        }

        self._save_chart(json_body)

    def save_chart_rolling_sharpe(self):
        num_periods = self.v.shape[0] - 1
        num_a = min(60, num_periods)
        num_b = min(252, num_periods)

        json_body = {
            "chart": {
                "title": "Rolling Sharpe evolution",
                "chartable_type": "Backtest",
                "chartable_id": self.backtest_id,
                "chart_traces": [
                    {
                        "x_name": "Dates",
                        "y_name": f"Sharpe: {num_a} periods",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(
                            self.sharpe_ratio_rolling(num_a)
                            .fillna(method="bfill")
                            .values
                        ),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": f"Sharpe: {num_b} periods",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(
                            self.sharpe_ratio_rolling(num_b)
                            .fillna(method="bfill")
                            .values
                        ),
                    },
                ],
            }
        }

        self._save_chart(json_body)

    def save_chart_historical_returns(self):
        num_periods = self.v.shape[0] - 1
        rolling_num_a = min(20, num_periods)
        rolling_num_b = min(60, num_periods)
        rolling_num_c = min(252, num_periods)

        json_body = {
            "chart": {
                "title": "Rolling return evolution",
                "chartable_type": "Backtest",
                "chartable_id": self.backtest_id,
                "chart_traces": [
                    {
                        "x_name": "Dates",
                        "y_name": f"{rolling_num_a} periods",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(
                            self.v.pct_change(periods=rolling_num_a)
                            .fillna(method="bfill")
                            .values
                        ),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": f"{rolling_num_b} periods",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(
                            self.v.pct_change(periods=rolling_num_b)
                            .fillna(method="bfill")
                            .values
                        ),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": f"{rolling_num_c} periods",
                        "x_values": [str(el) for el in self.v.index],
                        "y_values": list(
                            self.v.pct_change(periods=rolling_num_c)
                            .fillna(method="bfill")
                            .values
                        ),
                    },
                ],
            }
        }

        self._save_chart(json_body)

    def save_chart_historical_leverage(self):
        json_body = {
            "chart": {
                "title": "Leverage evolution",
                "chartable_type": "Backtest",
                "chartable_id": self.backtest_id,
                "chart_traces": [
                    {
                        "x_name": "Dates",
                        "y_name": "Long",
                        "x_values": [str(el) for el in self.long_leverage.index],
                        "y_values": list(self.long_leverage.values),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": "Short",
                        "x_values": [str(el) for el in self.short_leverage.index],
                        "y_values": list(self.short_leverage.values),
                    },
                ],
            }
        }

        self._save_chart(json_body)

    def save_cumulative_returns(self):
        json_body = {
            "chart": {
                "title": "Cumulative returns",
                "chartable_type": "Backtest",
                "chartable_id": self.backtest_id,
                "chart_traces": [
                    {
                        "x_name": "Dates",
                        "y_name": "Returns",
                        "x_values": [str(el) for el in self.cumulative_return.index],
                        "y_values": list(self.cumulative_return.values),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": "Long returns",
                        "x_values": [
                            str(el) for el in self.cumulative_return_long.index
                        ],
                        "y_values": list(self.cumulative_return_long.values),
                    },
                    {
                        "x_name": "Dates",
                        "y_name": "Short returns",
                        "x_values": [
                            str(el) for el in self.cumulative_return_short.index
                        ],
                        "y_values": list(self.cumulative_return_short.values),
                    },
                ],
            }
        }

        self._save_chart(json_body)

    def _save_chart(self, json_body):
        response = requests.post(
            f"{self.api_endpoint}/charts/create_or_update",
            headers=self.request_headers,
            json=json_body,
        )

        if (
            response.status_code // 100 == 2
        ):  # Check if the status code is in the 200 range
            chart_id = response.json().get("id")
            print(f"Chart {chart_id} saved.")
        else:
            print(f"Chart creation failed with status code: {response.status_code}")
