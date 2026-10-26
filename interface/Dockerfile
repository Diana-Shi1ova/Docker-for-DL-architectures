FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.title="Deep Learning Web Interface"

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]