FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

ENTRYPOINT ["agentic-rag"]
