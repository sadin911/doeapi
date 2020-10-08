FROM tensorflow/tensorflow:2.1.0-py3
LABEL maintainer='username'

WORKDIR /home

RUN apt-get update && apt-get install -y \
	graphviz\
	wget\
	vim\
	git\
	build-essential cmake

COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt





