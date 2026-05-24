FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
COPY backend/requirements.tcn.txt /app/requirements.tcn.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install -r /app/requirements.tcn.txt

COPY backend/app /app/app
COPY data /app/data

RUN mkdir -p /app/data/synthetic /app/data/processed /app/data/models/offline

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
