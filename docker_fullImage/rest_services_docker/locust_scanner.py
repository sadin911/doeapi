from locust import HttpLocust, TaskSet, task
import base64
import json
from PIL import Image
import io



class UserBehavior(TaskSet):
    def scanner(self):
        IMAGE_PATH = r"../4connerwithsegment/images/input/20200123_091611.jpg"
        vizin = Image.open(IMAGE_PATH)
        image = open(IMAGE_PATH, "rb").read()
        image_encoded = base64.b64encode(image)
        image_base64 = image_encoded.decode('utf-8')
        event_data = {'image': image_base64}
        self.client.post("/api/docscanner", json=event_data)
    
    @task(1)
    def index(self):
        self.scanner()

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait = 5000
    max_wait = 9000