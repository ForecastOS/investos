# Contributing to InvestOS

InvestOS welcomes community contributions!

## Developer environment

1. Install [Python 3](https://www.python.org/downloads/)
    * We support versions >= 3.9.
2. Install `poetry` following their [installation guide](https://python-poetry.org/docs/#installation).
3. Install the package with dev dependencies:

    ```sh
    poetry install --with dev
    ```

## Running the tests

```sh
poetry run pytest
```

## Building the documentation

TBD.

## Code formatting

This project uses [Black](https://black.readthedocs.io/en/stable/) to format its code.
Both Python files (`*.py`) and IPython notebooks (`*.ipynb`) are reformatted.

To reformat the codebase, run:

```sh
poetry run black .
```

Black is also supported by most editors, so you should be able to integrate it to your workflow.
