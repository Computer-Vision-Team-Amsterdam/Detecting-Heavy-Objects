#!/bin/bash
set -e

poetry run isort .
poetry run black .
poetry run mypy --ignore-missing-imports --config-file=.mypyrc .
poetry run pylint --jobs=0 --rcfile=.pylintrc *.py
#poetry run pytest -s --cov=panorama --cov-report html --cov-report term-missing
#(cd docs && poetry run make html)
