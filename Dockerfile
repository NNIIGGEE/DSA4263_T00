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

RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install numpy
RUN pip install tensorflow
RUN pip install sentence_transformers
RUN pip install umap-learn
RUN pip install nltk
RUN pip install hdbscan
RUN pip install sklearn
RUN pip install bertopic
RUN pip install umap-learn
RUN pip install gensim
RUN pip install collections
RUN pip install matplotlib
RUN pip isntall stop_words
RUN pip install pprint

COPY . .
# CMD ["python", "./python_test.py"]
# CMD ["python", "./python_test2.py"]
# CMD ["python", "./model/protofair_model.py"]
CMD ["python", "./src/train_SA.py"]