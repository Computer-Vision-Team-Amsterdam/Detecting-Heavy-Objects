#FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04 AS base-image
FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04 AS base-image

ENV APP_USER=www \
    # Suppress OS prompts
    DEBIAN_FRONTEND=noninteractive \
    # Poetry home directory
    POETRY_HOME="/opt/poetry" \
    # Suppress prompts from Poetry
    POETRY_NO_INTERACTION=1 \
    # Poetry version
    POETRY_VERSION=1.1.8 \
    # Create virtual environments in .venv directory in project root
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # Requirements and virtual environment directory
    PYSETUP_PATH=/opt/pysetup \
    VENV_PATH=/opt/pysetup/.venv \
    # Don't write .pyc files and don't buffer output to stdin/err
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Update path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/install-poetry.py | python

# Install project dependencies
# Detectron2 won't install with Poetry
WORKDIR $PYSETUP_PATH
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev \
    && poetry run python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Continue from clean Python base image
FROM base-image AS final

COPY --from=base-image $POETRY_HOME $POETRY_HOME
COPY --from=base-image $PYSETUP_PATH $PYSETUP_PATH
COPY . .
