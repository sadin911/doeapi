#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:08:18 2020

@author: chonlatid
"""

import os
import shutil
import tensorflow as tf
#tf.disable_v2_behavior()
import datetime
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

from base64 import b64decode
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import BatchNormalization,Dropout, MaxPooling2D , GlobalMaxPool1D
from tensorflow.python.keras.layers import Input,Activation, Dense, Flatten, Concatenate, LSTM, Embedding
import numpy as np
import gendata
import io
import sys
import glob2
from os.path import join
import tensorflow.keras.backend as K
import random
from tensorflow.compat.v1.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import set_session
import cv2
import imutils
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class DocScanner():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.gf = 16
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.model = self.build_segmodel()
        op = Adam(0.0001)
        self.model.compile(loss=['binary_crossentropy','mse'],
                              loss_weights=[100, 1],
                              optimizer=op)
        self.model.summary()
        self.gen_data = gendata.gendata()
    
    def build_segmodel(self):
        def conv2d(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum = 0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)

            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(d)
            d = BatchNormalization(momentum = 0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)

            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(d)
            d = BatchNormalization(momentum = 0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPooling2D()(d)
           
            return d
        
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum = 0.8)(u)
            u = Concatenate()([u, skip_input])
            
            return u
        
        def conv2dskip(layer_input,skip_input, filters, f_size=3):
            """Layers used during downsampling"""
            di0 = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            di0 = BatchNormalization(momentum = 0.8)(di0)
            di0 = LeakyReLU(alpha=0.2)(di0)

            di1 = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(di0)
            di1 = BatchNormalization(momentum = 0.8)(di1)
            di1 = LeakyReLU(alpha=0.2)(di1)

            di2 = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(di1)
            di2 = BatchNormalization(momentum = 0.8)(di2)
            di2 = LeakyReLU(alpha=0.2)(di2)
            di2 = MaxPooling2D()(di2)
            
            dx = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(skip_input)
            dx = MaxPooling2D()(dx)
            dy = Concatenate()([di2, dx])
            return dy
        
        # Image input
        d0 = Input(shape=self.img_shape)
        # Downsampling
        d1 = conv2d(d0, self.gf , 7)
        d2 = conv2d(d1, self.gf , 7)
        dp = Dropout(0.2)(d2)
        d3 = conv2d(dp, self.gf*2 , 5)
        d4 = conv2d(d3, self.gf*2 , 5)
        dp = Dropout(0.2)(d4)
        d5 = conv2d(dp, self.gf*4)
        d6 = conv2d(d5, self.gf*4)
        dp = Dropout(0.2)(d6)
        d7 = conv2d(dp, self.gf*8)
        
        u0 = deconv2d(d7,d6,self.gf*8*2)
        u1 = deconv2d(u0,d5,self.gf*4*2)
        u2 = deconv2d(u1,d4,self.gf*4*2,dropout_rate=0.2)
        u3 = deconv2d(u2,d3,self.gf*2*2)
        u4 = deconv2d(u3,d2,self.gf*2*2,dropout_rate=0.2)
        u5 = deconv2d(u4,d1,self.gf*2)
        u6 = deconv2d(u5,d0,self.gf*2,dropout_rate=0.2)
        
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u6)

        # Downsampling
        dd1 = conv2dskip(output_img,d0, self.gf , 7)
        dd2 = conv2dskip(dd1,d1, self.gf , 7)
        dd3 = conv2dskip(dd2,d2, self.gf*2, 5)
        dd4 = conv2dskip(dd3,d3, self.gf*2 ,5)
        dd5 = conv2dskip(dd4,d4, self.gf*4)
        dd6 = conv2dskip(dd5,d5, self.gf*4)


        f = Flatten()(dd6)
        dn = Dense(1024)(f)
        dn = LeakyReLU(alpha=0.2)(dn)
        dn = Dense(8)(dn)
        outcorner = Activation('linear')(dn)
        
        
        model = Model(
			inputs=d0,
			outputs=[output_img, outcorner])
        
        return model

    def test(self,pathinput):
        img_viz = Image.open(pathinput).convert('RGB')
        ratio_h = img_viz.height / self.img_rows
        ratio_w = img_viz.width / self.img_cols
        orgimg = np.asarray(img_viz)
        orgimg = orgimg/127.5 - 1
        
        img_viz = img_viz.resize((256,256))
        img_viz = np.asarray(img_viz)
        img_viz = img_viz/127.5 - 1
        
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)
        
        
        [predict_mask,predict_corner] = self.model.predict(indput_data)
        predict_corner = np.reshape(predict_corner[0],(4,2))
        predict_corner[:,0] *= ratio_w  
        predict_corner[:,1] *= ratio_h
        predict_corner = np.reshape(predict_corner,(8))
        viz_img = self.gen_data.plot_corner(orgimg,predict_corner)
        predict_mask = cv2.resize(predict_mask[0],(orgimg.shape[1],orgimg.shape[0]))
       

        warped = self.transformFourPoints((orgimg+1)*127.5, predict_corner.reshape(4, 2))
        warped = Image.fromarray(warped.astype('uint8'),'RGB')
        
        # predict_corner_mark = self.scanedge(predict_mask*255)
        # predict_corner_mark = np.reshape(predict_corner_mark,(8))
        # print(predict_corner_mark)
        # print(predict_corner)
        # viz_img_marked = self.gen_data.plot_corner(orgimg,predict_corner_mark)
        predict_seg = self.gen_data.plot_corner_seg(predict_mask,predict_corner)

        # [ 478.632 127.008 1247.736 129.36  1246.56 1148.952 1247.736  129.36]
        # [ 448.795 156.549 1242.905 151.70  1223.99 1064.216 462.0461  1078.34]

        # img_warpmark = self.transformFourPoints((orgimg+1)*127.5, predict_corner_mark.reshape(4, 2))
        # img_warpmark = Image.fromarray(img_warpmark.astype('uint8'),'RGB')
        try:
            predict_seg.save(os.path.join('masktest' + '.jpg'))
            viz_img.save(os.path.join('debugtest' + '.jpg'))
            # viz_img_marked.save(os.path.join('debugmark' + '.jpg'))
            warped.save(os.path.join('warpedtest' + '.jpg'))
            # img_warpmark.save(os.path.join('warped_mark' + '.jpg'))

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    def predictcorner(self,b64_input_image):
        byteImage = b64decode(b64_input_image)
        pilimg = Image.open(io.BytesIO(byteImage))
        pilimg = pilimg.resize((256,256))
        npimg = np.asarray(pilimg)
        orgimg = npimg.copy()
        npimg = npimg/127.2 - 1

        ratio_h = pilimg.height / self.img_rows
        ratio_w = pilimg.width / self.img_cols

        indput_data = npimg
        indput_data = np.expand_dims(indput_data, axis = 0)
        with self.graph.as_default():
            set_session(self.sess)
            [predict_mask,predict_corner] = self.model.predict(indput_data)

        predict_corner[:,0] *= ratio_w  
        predict_corner[:,1] *= ratio_h

        viz_img = self.gen_data.plot_corner(orgimg,predict_corner[0])
        predict_seg = self.gen_data.plot_corner_seg(predict_mask[0],predict_corner[0])
        img_warpcv = self.transformFourPoints()

    def order_points(self,pts):
    	rect = np.zeros((4, 2), dtype="float32")
    	s = pts.sum(axis=1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    	diff = np.diff(pts, axis=1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
    	return rect

    def transformFourPoints(self,image_cv, pts):
        
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
    
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
    
        return warped

    def scanedge(self,image):
        height_for_conversion = min(1000, image.shape[0])
        conversion_ratio = image.shape[0] / height_for_conversion
        image = imutils.resize(image, height = height_for_conversion)

        # Edge Detection

        gray = image
        gray = cv2.bilateralFilter(gray, 3, 75, 75)  # Tune parameters for better processing
        edged = cv2.Canny(gray.astype('uint8'), 75, 200)
        cv2.imwrite('edge.jpg',edged)
        # Find contours

        contours, hierarchy =cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:min(3, len(contours))]

        flag = 0

        for contour in contours:

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(polygon) == 4:
                flag = 1
                break

        if flag == 0:
            print("No contour found with 4 points")
            exit(1)

        # Change image perspective

        return  polygon * conversion_ratio
    
    
if __name__ == '__main__':
    if(os.path.isdir('logs')):
        shutil.rmtree('logs')
    doc = DocScanner()
    doc.model.load_weights(r'checkpoint_lr5_100_5_m250_r8_bg_revert.h5')
    doc.test('images/input/20200124_172559.jpg')
    # gan.train(epochs=1000000, batch_size=1, sample_interval=100)