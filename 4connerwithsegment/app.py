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
#global td

#global graph, service

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

@app.route("/api/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
#    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
#        if flask.request.files.get("image"):
#        data["image"] = (flask.request.json["image"])
        data = service.classify_doc(flask.request.json["image"])
            # read the image in PIL format
#            image = flask.request.files["image"].read()
#            image = Image.open(io.BytesIO(image))
#
#            # preprocess the image and prepare it for classification
#            image = prepare_image(image, target=(224, 224))
#
#            # classify the input image and then initialize the list
#            # of predictions to return to the client
#            preds = model.predict(image)
#            results = imagenet_utils.decode_predictions(preds)
#            data["predictions"] = []
#
#            # loop over the results and add them to the list of
#            # returned predictions
#            for (imagenetID, label, prob) in results[0]:
#                r = {"label": label, "probability": float(prob)}
#                data["predictions"].append(r)

            # indicate that the request was a success
#        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    global graph
    global service 
    service = DocClassifyService()
    app.run(host = '192.168.182.238',port=5005)
    # app.run(debug=True)
    
    