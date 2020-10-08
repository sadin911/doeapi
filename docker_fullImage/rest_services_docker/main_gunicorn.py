import flask
from flask import Flask
from flask import request, make_response, jsonify
from services.DocScanner import DocScanner
from services.faceCheckAndCrop import FaceCheck
from services.FaceMatchService import FaceMatchService
from services.DocClassifyService import DocClassifyService
import base64
import cv2
import json
import io
from PIL import Image
from services.FaceAlignment import faceAlignment
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import gc
import os
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
app = Flask(__name__)
doc = DocScanner()
face = FaceCheck()
facematch = FaceMatchService()
docclass = DocClassifyService()
facealign = faceAlignment()
# cache = Cache(app,config={'CACHE_TYPE': 'simple',"CACHE_DEFAULT_TIMEOUT": 50})
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per second"]
)

@app.route('/')
def root():
    return "pyService is running !"

@app.route("/api/doccorner", methods=["POST"])
@limiter.limit("1000000 per second")
def doccorner():
    if flask.request.method == "POST":
        image_base64 = flask.request.json["image"]
        width = flask.request.json["width"]
        height = flask.request.json["height"]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
        corner = doc.onlycorner(image,width,height)
        print('success')
        gc.collect()
    return flask.jsonify({"corner":corner,"isSuccessful" : True})

@app.route("/api/docscanner", methods=["POST"])
@limiter.limit("1000000 per second")
def docscaner():
    try:
        if flask.request.method == "POST":
            image_base64 = flask.request.json["image"]
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
            warpimage = doc.predictcorner(image)
            success, warpimage_cv = cv2.imencode('.jpg', warpimage)
            warpimage_str = warpimage_cv.tostring()
            warpimage_encoded = base64.b64encode(warpimage_str)
            warpimage_base64 = warpimage_encoded.decode('utf-8')
            print('success')

        return flask.jsonify({"image" : warpimage_base64,
        "isSuccessful" : True})

    except Exception as e: 
        print('error')
        return flask.jsonify({"message" : e ,
        "isSuccessful" : False})

@app.route("/api/avatar", methods=["POST"])
@limiter.limit("1000000 per second")
def facechecking():
    try:
        if flask.request.method == "POST":
            image_base64 = flask.request.json["image"]
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
            response = face.process(image)
        return flask.jsonify(response)

    except Exception as e: 
        return flask.jsonify({"message" : e ,
        "isSuccessful" : False})

@app.route("/api/facealign", methods=["POST"])
@limiter.limit("1000000 per second")
def facealign():
    if flask.request.method == "POST":
        image_base64 = flask.request.json["image"]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
        warpimage = facealign.face_alignment(image)
        success, warpimage_cv = cv2.imencode('.jpg', warpimage)
        warpimage_str = warpimage_cv.tostring()
        warpimage_encoded = base64.b64encode(warpimage_str)
        warpimage_base64 = warpimage_encoded.decode('utf-8')
    return flask.jsonify({"img" : warpimage_base64,
    "isSuccessful" : True})

@app.route("/api/facematch", methods=["POST"])
@limiter.limit("1000000 per second")
def facematch():
    if flask.request.method == "POST":
        image1b64 = flask.request.json["image1"]
        image1 = Image.open(io.BytesIO(base64.b64decode(image1b64))).convert('RGB')
        image2b64 = flask.request.json["image2"]
        image2 = Image.open(io.BytesIO(base64.b64decode(image2b64))).convert('RGB')
        data = facematch.face_match(image1,image2)
        # print(data)
    return flask.jsonify(data)

@app.route("/api/docclassify", methods=["POST"])
@limiter.limit("1000000 per second")
def docclassify():
    if flask.request.method == "POST":
        data = docclass.classify_doc(flask.request.json["image"])
        print(data)
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080, threaded=True)