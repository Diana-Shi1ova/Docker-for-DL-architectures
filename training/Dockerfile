FROM docker_practice-inference:latest

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY train.py /app/train.py

CMD ["python", "/app/train.py"]