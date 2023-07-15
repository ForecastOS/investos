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

You can use [pre-commit hooks](#pre-commit-hooks) to ensure your changes are formatted properly.

Black is also supported by most editors, so you should be able to integrate it to your workflow.

## Pre-commit hooks

This repo uses [pre-commit](https://pre-commit.com/) for contributors to be able to quickly validate their changes.

To enable this:

1. Install `pre-commit` following their [installation guide](https://pre-commit.com/#install)
2. Install the hooks for this repository:

    ```sh
    pre-commit install
    ```

You will now have sanity checks running before every commit, only on your changes.

You can also run the hooks on all files with:

```sh
pre-commit run --all-files
```
