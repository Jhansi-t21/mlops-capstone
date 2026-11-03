# ---- Step 8: Docker image for the prediction API ----
# Build (needs models/model.joblib, which `dvc repro` creates):
#   docker build -t breast-cancer-api:latest .
# Run:
#   docker run -d --name bc-api -p 8000:8000 breast-cancer-api:latest
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# 1) install dependencies first (cached unless requirements-api.txt changes)
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) copy only what the API needs at runtime
COPY src/ ./src/
COPY app/ ./app/
COPY models/model.joblib models/model_info.json ./models/

# 3) run as a non-root user and keep prediction logs in a writable folder
RUN useradd --create-home appuser && mkdir -p /code/logs && chown -R appuser /code
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
