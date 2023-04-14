FROM python:3.9-slim-buster

WORKDIR /DSA4263_T00

COPY . .
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --ignore-installed
EXPOSE 5000

ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=5000"] 