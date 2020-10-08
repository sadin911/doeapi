from locust import HttpLocust, TaskSet, task
import base64
import json
from PIL import Image
import io



class UserBehavior(TaskSet):
    def scanner(self):
        IMAGE_PATH = r'./image/Baifern/37156239_417982962062250_8698652647259203198_n.jpg'
        vizin = Image.open(IMAGE_PATH)
        image = open(IMAGE_PATH, "rb").read()
        image_encoded = base64.b64encode(image)
        image_base64 = image_encoded.decode('utf-8')
        event_data = {'image1': image_base64,'image2': image_base64}
        self.client.post("/api/facematch", json=event_data)
    
    @task(1)
    def index(self):
        self.scanner()

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait = 5000
    max_wait = 9000