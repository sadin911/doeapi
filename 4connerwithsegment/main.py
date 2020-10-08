import flask
from flask import Flask
from flask import request, make_response, jsonify
from DocScanner import DocScanner
import base64
import cv2
import json
app = Flask(__name__)


@app.route('/')
def root():
    return "DocScanner is Running !"

@app.route("/api/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        warpimage = service.predictcorner(flask.request.json["image"])
        success, warpimage_cv = cv2.imencode('.jpg', warpimage)
        warpimage_str = warpimage_cv.tostring()
        warpimage_encoded = base64.b64encode(warpimage_str)
        warpimage_base64 = warpimage_encoded.decode('utf-8')
    return flask.jsonify({"img" : warpimage_base64,
    "isSuccessful" : True})

@app.route("/api/facecrop", methods=["POST"])
def facecrop():
    if flask.request.method == "POST":
        warpimage = service.predictcorner(flask.request.json["image"])
        success, warpimage_cv = cv2.imencode('.jpg', warpimage)
        warpimage_str = warpimage_cv.tostring()
        warpimage_encoded = base64.b64encode(warpimage_str)
        warpimage_base64 = warpimage_encoded.decode('utf-8')
    return flask.jsonify({"img" : warpimage_base64,
    "isSuccessful" : True})

if __name__ == '__main__':
    service = DocScanner()
    app.run(host='127.0.0.1', port=8080)
    # app.run(debug=True)