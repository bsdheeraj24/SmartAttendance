FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAKEFLAGS=-j1

# Build dependencies required when dlib is compiled from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "server.py"]
