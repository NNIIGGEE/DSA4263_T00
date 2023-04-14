FROM python:3.9-slim-buster

WORKDIR /DSA4263_T00

COPY . .
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
EXPOSE 5001

ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=5001"] 