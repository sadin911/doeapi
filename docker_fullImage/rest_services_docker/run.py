from helper import *
import FaceAlignment
from PIL import Image
import cv2
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    img = Image.open('./image/tilt/smth.png').convert('RGB')
    img=FaceAlignment.face_alignment(img,1)
    cv2.imwrite('testalign.jpg',img)
