FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    libffi-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000
CMD ["python", "app.py"]
