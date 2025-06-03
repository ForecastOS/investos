import investos.portfolio

__version__ = "0.6.0"

import os

api_key = os.environ.get("FORECASTOS_API_KEY", "")
api_endpoint = os.environ.get(
    "FORECASTOS_API_ENDPOINT", "https://app.forecastos.com/api/v1"
)
