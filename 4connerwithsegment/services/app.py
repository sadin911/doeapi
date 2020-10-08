#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:38:24 2020

@author: trainai
"""
import flask
from flask import Flask
from flask import request, make_response, jsonify
from DocClassifyService import DocClassifyService
from tensorflow.python.keras.backend import set_session
import json

#global td


service = DocClassifyService()
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    
    if flask.request.method == "POST":
        data = service.classify_doc(flask.request.json["image"])

    return flask.jsonify(data)


if __name__ == '__main__':

    app.run(debug=True,host='0.0.0.0',port='5006')
    
    