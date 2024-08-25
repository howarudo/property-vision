FROM python:3.10-slim

RUN pip install poetry==1.5.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONUNBUFFERED=1


WORKDIR /app

COPY poetry.lock pyproject.toml ./
COPY src ./src
COPY data ./data
RUN touch README.md

RUN poetry install --no-dev --no-root

CMD ["poetry", "run", "python", "-u", "src/main.py"]
