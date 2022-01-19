#!/bin/bash
set -e

#isort .
#black .
mypy --ignore-missing-imports --config-file=.mypyrc .
pylint --jobs=0 --rcfile=.pylintrc *.py --disable=W,C,R evaluation.py
pytest -s --cov=. --cov-report html --cov-report term-missing
