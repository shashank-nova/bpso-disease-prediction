FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --upgrade pip && pip install '.[serve]'
COPY app.py ./app.py
COPY assets ./assets
EXPOSE 8000 8501
CMD ["uvicorn", "bpso_fs.api:app", "--host", "0.0.0.0", "--port", "8000"]
