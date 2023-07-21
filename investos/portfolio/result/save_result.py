import requests


class SaveResult:
    def save(
        self,
        description,
        tags,
        api_key,
        api_endpoint="https://app.forecastos.com/api/v1",
    ):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.tags = tags

        self.save_backtest(description)
        self.save_backtest_charts()

    def save_backtest(self, description):
        json_body = {
            "backtest": {
                "description": description,
                "performance_summary": {
                    "total_return": self.total_return,
                    "annualized_return": self.annualized_return,
                    "annualized_excess_return": self.annualized_excess_return,
                    "sharpe_ratio": self.sharpe_ratio,
                    "max_drawdown": self.max_drawdown,
                    "annual_turnover": self.annual_turnover,
                },
                "portfolio_id": None,
                "team_id": None,
                "project_id": None,
                "tags": self.tags,
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

        response = requests.post(
            f"{self.api_endpoint}/charts", headers=self.request_headers, json=json_body
        )

        if (
            response.status_code // 100 == 2
        ):  # Check if the status code is in the 200 range
            chart_id = response.json().get("id")
            print(f"Chart {chart_id} saved.")
        else:
            print(f"Chart creation failed with status code: {response.status_code}")

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

        response = requests.post(
            f"{self.api_endpoint}/charts", headers=self.request_headers, json=json_body
        )

        if (
            response.status_code // 100 == 2
        ):  # Check if the status code is in the 200 range
            chart_id = response.json().get("id")
            print(f"Chart {chart_id} saved.")
        else:
            print(f"Chart creation failed with status code: {response.status_code}")
