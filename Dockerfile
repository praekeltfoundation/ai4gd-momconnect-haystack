FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync

COPY . .

CMD ["uv", "run", "uvicorn", "ai4gd_momconnect_haystack.api:app", "--host", "0.0.0.0", "--port", "8000"]
