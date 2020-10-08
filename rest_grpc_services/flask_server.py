import flask
from flask import Flask
from flask import request, make_response, jsonify
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import base64
import cv2
import json
import io
from PIL import Image
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tensorflow as tf
import numpy as np
tf.compat.v1.app.flags.DEFINE_string('server', '127.0.0.1:8500',
                                    'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS
channel = grpc.insecure_channel(FLAGS.server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'corner'

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["300 per second"]
)

@app.route('/')
def root():
    return "pyService is running !"

@app.route("/api/doccorner", methods=["POST"])
@limiter.limit("30 per second")
def doccorner():
    if flask.request.method == "POST":
        image_base64 = flask.request.json["image"]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
        image = image.resize((256,256))
        image = np.asarray(image).astype('float32')/127.5 - 1 
        image = np.expand_dims(image, axis = 0)
        buffer = tf.make_tensor_proto(image, shape=[1,256,256,3])
        request.inputs['input_1'].CopyFrom(buffer)
        result = stub.Predict(request)
        print(result)
    return flask.jsonify({"output":str(result),"isSuccessful" : True})

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080, threaded=True)