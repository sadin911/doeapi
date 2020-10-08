
from base64 import b64decode
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model


class DocClassifyService:
    def __init__(self):
        self.model_path = r'doc_classify_v0.3.h5'
        
        # tf_config = some_custom_config
        self.model = load_model(self.model_path)
        self.model_input_size = (256,256)
        self.class_label = ['cid','passport','house registration','niti', 'unknown']
        
    def classify_doc(self,b64_input_image):
        try:
            byteImage = b64decode(b64_input_image)
            img = Image.open(io.BytesIO(byteImage))
            img = img.resize(self.model_input_size)
            img = np.asarray(img).astype('float')
            img = (img/127.5)-1
            img = np.expand_dims(img,0)
    
            
            score = self.model.predict(img)[0]
            max_index = np.argmax(score)
            predict_class = self.class_label[max_index]
            
            # result = []
            # result.append({ 'score' : float(score[max_index])})
            # result.append({ 'predicted_class' : predict_class})
            # result.append({ 'all_class_info' : self.class_label})
            # result.append({ 'isSuccessful' : True})
            
            result = {
                'score' : float(score[max_index]),
                'predicted_class' : predict_class,
                'all_class_info' : self.class_label,
                'isSuccessful' : True
                }
        except Exception as e: 
            # result = []
            # result.append({ 'isSuccessful' : True})
            # result.append({ 'error' : e})
            reusult = {
                'isSuccessful' : False,
                'error' : e
                }
        
        # result = {'score':score[max_index],
        #           'class': predict_class,
        #           'class_info' : self.class_label,
        #           'isSuccessful': True
        #           }
        return result
