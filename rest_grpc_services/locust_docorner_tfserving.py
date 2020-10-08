from locust import HttpLocust, TaskSet, task,between,events
import base64
import json
from PIL import Image
import io
import numpy as np
import os
import glob
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import sys
import threading
import grpc
import tensorflow as tf
import time
IMAGE_PATH = r"../dataset/docdata/20200204_101758.jpg"
vizin = Image.open(IMAGE_PATH)
height = vizin.height
vizin = vizin.resize((256,256))
image = np.asarray(vizin).astype('float32')/127.5 - 1 
image = np.expand_dims(image, axis = 0)
rawBytes = io.BytesIO()
vizin.save(rawBytes, "jpeg")
rawBytes.seek(0)
image_encoded = base64.b64encode(rawBytes.read())
image_base64 = image_encoded.decode('utf-8')

channel = grpc.insecure_channel('127.0.0.1:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'corner'
request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(image, shape=[1,256,256,3]))

class UserBehavior(TaskSet):
    @task(1)
    def corner(self):
        IMAGE_PATH = r"../dataset/docdata/20200204_101758.jpg"
        vizin = Image.open(IMAGE_PATH)
        image = open(IMAGE_PATH, "rb").read()
        image_encoded = base64.b64encode(image)
        image_base64 = image_encoded.decode('utf-8')
        event_data = {'image': image_base64}
        self.client.post("api/doccorner", json=event_data)
    
            
    def index(self):
        self.corner()

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    wait_time = between(5, 9)