FROM python:3.10-bookworm
COPY requirements.txt /requirements.txt
RUN  apt-get update && apt-get upgrade -y && apt-get install cmake -y && apt install libopenblas-dev -y
RUN pip install -r requirements.txt

