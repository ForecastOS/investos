import os
import warnings

import forecastos as fos

import investos.portfolio

__version__ = "0.10.0"

fos.api_key = os.environ.get("FORECASTOS_API_KEY", "")
fos.api_key_team = os.environ.get("FORECASTOS_API_KEY_TEAM", "")
fos.api_endpoint = "https://app.forecastos.com/api/v1"

warnings.warn(
    "The 'investos' package has been integrated into and is now "
    "actively maintained under the 'forecastos' package by the same team.\n\n"
    "Please install and use:\n"
    "    pip install forecastos\n\n"
    "Docs: https://forecastos.com/guides/introduction/getting_started\n"
    "PyPI: https://pypi.org/project/forecastos",
    UserWarning,
    stacklevel=2,
)
