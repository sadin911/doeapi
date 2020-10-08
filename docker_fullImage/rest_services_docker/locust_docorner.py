from locust import HttpLocust, TaskSet, task,between,events
import base64
import json
from PIL import Image
import io
import numpy as np
import os
import glob
import sys
import tensorflow as tf
import time
IMAGE_PATH = r"../dataset/docdata/20200204_101758.jpg"
vizin = Image.open(IMAGE_PATH)
height = vizin.height
width = vizin.width
vizin = vizin.resize((256,256))
image = np.asarray(vizin).astype('float32')/127.5 - 1 
image = np.expand_dims(image, axis = 0)
rawBytes = io.BytesIO()
vizin.save(rawBytes, "jpeg")
rawBytes.seek(0)
image_encoded = base64.b64encode(rawBytes.read())
image_base64 = image_encoded.decode('utf-8')

class UserBehavior(TaskSet):
    @task(1)
    def corner(self):
        event_data = {'image': image_base64,'width':width,'height':height}
        self.client.post("/api/doccorner", json=event_data)
            
    def index(self):
        self.corner()

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    wait_time = between(5, 9)