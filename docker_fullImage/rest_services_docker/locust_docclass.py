from locust import HttpLocust, TaskSet, task,between
import base64
import json
from PIL import Image
import io



class UserBehavior(TaskSet):
    def scanner(self):
        IMAGE_PATH = r'/home/chonlatid/Python/Project/docsegmentation/docker_fullImage/rest_services_docker/image/facematch/download (1).jpeg'
        vizin = Image.open(IMAGE_PATH)
        image = open(IMAGE_PATH, "rb").read()
        image_encoded = base64.b64encode(image)
        image_base64 = image_encoded.decode('utf-8')
        event_data = {'image': image_base64}
        self.client.post("api/docclassify", json=event_data,verify=False)
    
    @task(1)
    def index(self):
        self.scanner()

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    wait_time = between(5,9)