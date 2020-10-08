#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:08:18 2020

@author: chonlatid
"""
import tensorflow as tf
import shutil

import datetime
from PIL import Image

import base64
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import BatchNormalization,Dropout, MaxPooling2D , GlobalMaxPool1D
from tensorflow.python.keras.layers import Input,Activation, Dense, Flatten, Concatenate, LSTM, Embedding
import numpy as np
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

class DocScanner():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.gf = 16
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.model = load_model('corner.h5')

    def predictcorner(self,img_viz):
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
        
        predict_mask = cv2.resize(predict_mask[0],(orgimg.shape[1],orgimg.shape[0]))
        warped = self.transformFourPoints((orgimg+1)*127.5, predict_corner.reshape(4, 2))
        warped = Image.fromarray(warped.astype('uint8'),'RGB')
        return np.asarray(warped)

    def onlycorner(self,img_viz,width,height):
        ratio_h = height / self.img_rows
        ratio_w = width / self.img_cols
        orgimg = np.asarray(img_viz).astype('float')
        orgimg = orgimg/127.5 - 1

        img_viz = np.asarray(img_viz)
        img_viz = img_viz/127.5 - 1
        
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)
        
        [predict_mask,predict_corner] = self.model.predict(indput_data)
        predict_corner = np.reshape(predict_corner[0],(4,2))
        predict_corner[:,0] *= ratio_w  
        predict_corner[:,1] *= ratio_h
        predict_corner = np.reshape(predict_corner,(8))

        return str(predict_corner)

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