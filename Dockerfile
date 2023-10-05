FROM python:3.8-slim

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/poetry'
    ENV=PROD


RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && curl -sSL https://install.python-poetry.org | python - \
    && mv $POETRY_HOME/bin/poetry /usr/local/bin \
    && apt-get purge -y --auto-remove curl


RUN mkdir -p /fraud-detection-system
WORKDIR /fraud-detection-system

COPY ./pyproject.toml ./poetry.lock* ./

COPY . .

# Make the script executable
RUN chmod +x docker-entrypoint.sh

RUN poetry install -vvv --no-root

# Set our script as the default entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]