FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./

# RUN uv init
RUN uv venv
RUN uv pip install .
RUN uv sync

COPY . .

CMD ["uv", "run", "main.py"]
