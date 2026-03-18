FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --only-binary=:all: dlib==19.24.6 \
    && pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "server.py"]
