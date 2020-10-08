import flask
from flask import Flask
from flask import request, make_response, jsonify
from services.DocScanner import DocScanner
from services.faceCheckAndCrop import FaceCheck
from services.FaceMatchService import FaceMatchService
from services.DocClassifyService import DocClassifyService
from services.nameListService import NameListService
import services.FaceAlignment as faceAlignment
import tensorflow as tf
import base64
import cv2
import json
import io
from PIL import Image

from passporteye.mrz.image import MRZPipeline
from passporteye import read_mrz


import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import os
from memory_profiler import memory_usage

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["300 per second"]
)

doc = DocScanner()
face = FaceCheck()
facematcher = FaceMatchService()
docclass = DocClassifyService()
facealign = faceAlignment.faceAlignment()
nameListOcr = NameListService()
@app.route('/')
def root():
    return "pyService is running !"

@app.route("/api/doccorner", methods=["POST"])
def doccorner():
    if flask.request.method == "POST":
        image_base64 = flask.request.json["image"]
        width = flask.request.json["width"]
        height = flask.request.json["height"]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
        corner = doc.onlycorner(image,width,height)
        print('success')
    return flask.jsonify({"corner":corner,"isSuccessful" : True})

@app.route("/api/docscanner", methods=["POST"])
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

@app.route("/api/facecrop", methods=["POST"])
def facecrop():
    try:
        if flask.request.method == "POST":
            image_base64 = flask.request.json["image"]
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
            # warpimage = facealign.face_alignment(image)
            response = face.process_noResize(image)
        return flask.jsonify(response)

    except Exception as e: 
        return flask.jsonify({"message" : e ,
        "isSuccessful" : False})

@app.route("/api/facealign", methods=["POST"])
def facealignment():
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
def facematch():
    if flask.request.method == "POST":
        image1b64 = flask.request.json["image1"]
        image1 = Image.open(io.BytesIO(base64.b64decode(image1b64))).convert('RGB')
        image2b64 = flask.request.json["image2"]
        image2 = Image.open(io.BytesIO(base64.b64decode(image2b64))).convert('RGB')
        data = facematcher.face_match(image1,image2)
        # print(data)
    return flask.jsonify(data)

@app.route("/api/docclassify", methods=["POST"])
def docclassify():
    if flask.request.method == "POST":
        data = docclass.classify_doc(flask.request.json["image"])
        print(data)
    return flask.jsonify(data)

@app.route("/api/datapage", methods=["POST"])
def datapage():
    if flask.request.method == "POST":
        image1b64 = flask.request.json["image"]
        image1 = io.BytesIO(base64.b64decode(image1b64))
        p = MRZPipeline(image1, extra_cmdline_params='-l ocrb --oem 3 --psm 3')
        mrz = p.result.to_dict()
        json_object = json.dumps(mrz) 
        result = ({
                'isSuccessful' : True,
                'mrz':json_object
                })
        # print(data)
    return flask.jsonify(result)

@app.route("/api/namelist", methods=["POST"])
def namelist():
    if flask.request.method == "POST":
        image_base64 = flask.request.json["image"]
        # img = Image.open(io.BytesIO(base64.b64decode(image1b64)))
        # mage_base64 = flask.request.json["image"]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
        warpimage = doc.predictcorner(image)
        # success, warpimage_cv = cv2.imencode('.jpg', warpimage)
        # print(warpimage)
        nameList,passportList = nameListOcr.nameList(warpimage)
        result = ({
                'isSuccessful' : True,
                'nameList':nameList,
                'passportList':passportList
                })
        # print(data)
    return flask.jsonify(result)

@app.route("/api/namelistcropped", methods=["POST"])
def namelistcropped():
    if flask.request.method == "POST":
        image1b64 = flask.request.json["image"]
        img = Image.open(io.BytesIO(base64.b64decode(image1b64)))
        print(img)
        nameList,passportList = nameListOcr.nameListCropped(img)
        result = ({
                'isSuccessful' : True,
                'nameList':nameList,
                'passportList':passportList
                })
        # print(data)
    return flask.jsonify(result)

if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    if tf.test.gpu_device_name():
        print('GPU')
    else:
        print("CPU")
    app.run(host = '0.0.0.0',port=8080, threaded=True)