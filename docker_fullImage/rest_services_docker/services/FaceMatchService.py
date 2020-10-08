# -*- coding: utf-8 -*-

import face_recognition
from PIL import Image
import io
import numpy as np

class FaceMatchService:
    def __init__(self):
        print('init')
    def face_match(self,img1,img2):
        try:
            img1_encoding = face_recognition.face_encodings(np.asarray(img1))[0]
            img2_encoding = face_recognition.face_encodings(np.asarray(img2))[0]
    
            score = 1 - face_recognition.face_distance([img1_encoding], img2_encoding)[0]
            if(score>0.4):
                matched = True
            else:
                matched = False
                
            result = ({
            "score" : score,
            "matched" : matched,
            "isSuccessful" : True,
            })

        except Exception as e: 

            result = ({
                'isSuccessful' : False,
                'error' : e
                })

        print(type(result))
        return result
    
# if __name__ == '__main__':
#     from PIL import Image
#     img1 = Image.open('7702ddf3763832a0ef4ed9094b27ca40.jpg')
#     img2 = Image.open('f5df2b61e3310dad9c28e75b760a82b4.jpg')
#     service = FaceMatchService()
#     print(service.face_match(img1,img2))