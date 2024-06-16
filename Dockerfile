FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

EXPOSE 9092

CMD ["python", "kafka-saves3.py"]
