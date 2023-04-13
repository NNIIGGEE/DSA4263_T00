FROM python:3.9-buster

# RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev
RUN apt-get update
RUN apt-get -y upgrade
# RUN apt-update && 
# RUN apt-upgrade
RUN apt install build-essential
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get install -y python3-dev build-essential
RUN pip install --upgrade pip setuptools wheel
RUN pip install absl-py
RUN pip install --upgrade pip
COPY . src/
EXPOSE 5000
WORKDIR src/
RUN pip install --upgrade pip
RUN pip install --default-timeout=300 future
RUN pip install gensim
RUN pip install numpy==1.19.5
RUN pip install tensorflow
#RUN pip install bertopic --no-dependencies
RUN pip install -r requirements.txt
RUN pip install flask
RUN pip install flasgger
RUN pip install redis
#CMD python predict_api.py
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "src.app:app"]
#docker run -p 4000:5000 rf-new