# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:58:38 2019

@author: chonlatid.d
"""

# import the necessary packages
import requests
import base64
import json
import glob
import os
import time
import numpy as np
from PIL import Image
import io
# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = r"http://35.247.190.17:5005/api/predict"
IMAGE_PATH = r"images/input/20200123_091611.jpg"

# load the input image and construct the payload for the request
file_list = sorted(glob.glob(r'images/input/*.jpg'))
timelist = []
for i in range(len(file_list)):
    IMAGE_PATH = file_list[i]
    vizin = Image.open(IMAGE_PATH)
    image = open(IMAGE_PATH, "rb").read()
    image_encoded = base64.b64encode(image)
    image_base64 = image_encoded.decode('utf-8')
    event_data = {'image': image_base64}
    # submit the request
    start = time.time()
    res = requests.post(KERAS_REST_API_URL, verify = False , json=event_data)
    end = time.time()
    toc = end-start
    print("time = " + str(toc))
    timelist.append(toc)
    print("avg = " + str(np.average(timelist)))
    
    out = json.loads(res.content)
    image64 = out['img']
    image = Image.open(io.BytesIO(base64.b64decode(image64)))
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))
    #del out['face_base64']
    os.makedirs('jsonfromAPI', exist_ok=True)
    filesave = os.path.join('jsonfromAPI' , os.path.basename(file_list[i])[0:-4] + str('.json'))
    imgsave = os.path.join('jsonfromAPI' , os.path.basename(file_list[i])[0:-4] + str('zc.jpg'))
    vizsave = os.path.join('jsonfromAPI' , os.path.basename(file_list[i])[0:-4] + str('in.jpg'))
    with open(filesave, 'w') as data_file:
        out = json.dump(out, data_file)
        image.save(imgsave)
        vizin.save(vizsave)
        