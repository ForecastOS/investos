<h1>Getting Started</h1>

Welcome to the InvestOS portfolio engineering and backtesting framework!

InvestOS is an opinionated framework for constructing and backtesting portfolios in a consistent, albeit flexible way. We built it to make institutional-grade backtesting and portfolio optimization simple, extensible, and open-source.

This guide covers getting up and running with InvestOS.

Let's jump in!

## Prerequisites

**To run InvestOS you'll need**:

-   [Python +3.8](https://www.python.org/doc/)
    -   You can [download it here](https://www.python.org/downloads/)
    -   If you're working on MacOS, you may wish to [install it via Homebrew](https://docs.python-guide.org/starting/install3/osx/)
-   [pip](https://packaging.python.org/en/latest/key_projects/#pip)
    -   For installing InvestOS (and any other Python packages)
    -   [pip installation instructions here](https://packaging.python.org/en/latest/tutorials/installing-packages/)

**Although not required, running InvestOS might be easier if you have**:

-   [Poetry](https://python-poetry.org/), a package and dependency manager
-   Familiarity with [pandas](https://pandas.pydata.org/)
    -   The popular Python data analysis package (originally) released by AQR Capital Management

## Installation

If you're using pip:

```bash
$ pip install investos
```

If you're using poetry:

```bash
$ poetry add investos
```

## Importing InvestOS

At the top of your python file or .ipynb, add:

```python
import investos as inv
```

## Next: How InvestOS Works

Congratulations on setting up InvestOS!

Let's move on to our next guide: [How InvestOS Works](/guides/introduction/how_investos_works).
