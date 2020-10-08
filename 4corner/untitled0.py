# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 08:58:24 2018

@author: watcharapong.c
"""

import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import cv2
import io
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform

from PIL import ImageDraw




image_PATH = r'F:\Project\IDCardOCR\4corner detection\data\*.jpg'


save_PATH = r'F:\Project\IDCardOCR\4corner detection\gen_data'

impath = glob.glob(image_PATH)

pad_param = 40


for path in impath:
    
    img = Image.open(path)
    img = np.asarray(img)
#    img = img /255
    
    loc_topleft_img = np.zeros((img.shape[0],img.shape[1],3))
    loc_topleft_img[0][0]=255
    
    loc_bottomleft_img = np.zeros((img.shape[0],img.shape[1],3))
    loc_bottomleft_img[-1][0]=255
    
    loc_topright_img = np.zeros((img.shape[0],img.shape[1],3))
    loc_topright_img[0][-1]=255
    
    loc_bottomright_img = np.zeros((img.shape[0],img.shape[1],3))
    loc_bottomright_img[-1][-1]=255
    
    array_image = [loc_topleft_img,loc_bottomleft_img,loc_topright_img,loc_bottomright_img ]
    
    
    
    
    pad_top = int(abs(np.random.normal(0,pad_param)))
    pad_bottom = int(abs(np.random.normal(0,pad_param)))
    pad_left = int(abs(np.random.normal(0,pad_param)))
    pad_right = int(abs(np.random.normal(0,pad_param)))
 
    rotate_param = np.random.uniform(-30,30)
    

    
    img = Image.fromarray(img.astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
    img = cv2.copyMakeBorder( np.asarray(img), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
    
    for i in range(len(array_image)):
            
            ## rotate 
        img2 = Image.fromarray(array_image[i].astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (0,0,0))
        
        # pad border
        img3 = cv2.copyMakeBorder( np.asarray(img2), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(0,0,0))

        array_image[i] = img3
    
    ind = np.unravel_index(np.argmax(array_image[0], axis=None), array_image[1].shape)
    y_topleft = ind[0]
    x_topleft = ind[1]
    ind = np.unravel_index(np.argmax(array_image[1], axis=None), array_image[1].shape)
    y_bottomleft = ind[0]
    x_bottomleft = ind[1]
    ind = np.unravel_index(np.argmax(array_image[2], axis=None), array_image[1].shape)
    y_topright = ind[0]
    x_topright = ind[1]
    ind = np.unravel_index(np.argmax(array_image[3], axis=None), array_image[1].shape)
    y_bottomright = ind[0]
    x_bottomright = ind[1]
    
    borderimg=Image.fromarray(img)
    draw = ImageDraw.Draw(borderimg)
    draw.polygon([(x_topleft,y_topleft),(x_bottomleft,y_bottomleft),(x_bottomright,y_bottomright),(x_topright,y_topright) ],  outline=(255,0,0))
    
    
    fname = os.path.splitext(os.path.basename(path))[0]
    text_path = os.path.join(save_PATH,'txt',fname+'.txt')
    file = open(text_path,'w') 
 
    file.write(str(x_topleft)+ '\n' +  str(y_topleft) + '\n')
    file.write(str(x_bottomleft)+ '\n' +  str(y_bottomleft) + '\n')
    file.write(str(x_topright)+ '\n' +  str(y_topright) + '\n')
    file.write(str(x_bottomright)+ '\n' +  str(y_bottomright) + '\n')
    

    file.close() 
    
    
    borderimg.save(os.path.join(save_PATH,'check_img',os.path.basename(path)))
    img=Image.fromarray(img)
    img.save(os.path.join(save_PATH,'img',os.path.basename(path)))
    print(path)
    
   
    