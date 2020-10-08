# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:01:32 2018

@author: watcharapong.c
"""
import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation , LeakyReLU, Flatten, BatchNormalization, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.layers.recurrent import GRU 
from keras.optimizers import SGD, Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import glob
from PIL import Image,ImageDraw, ImageEnhance
import numpy as np
import cv2
import os
import io
import sys





path_input = r'C:\Users\trainai\Documents\watcharapong\4corner detection\data\*.jpg'
bg_path = r'C:\Users\trainai\Documents\watcharapong\trainBackground\*.jpg'
save_path = r'C:\Users\trainai\Documents\watcharapong\4corner detection\train'

#CURSOR_UP_ONE = '\x1b[1A'
#ERASE_LINE = '\x1b[2K'

class corner:
    def __init__(self):
        self.input_shape = (256,256,3)
        self.filter_count = 32
        self.kernel_size = (3, 3)
        self.leakrelu_alpha = 0.2
        self.model = None
        self.pathlist = glob.glob(path_input)
        self.pathbglist = glob.glob(bg_path)
        self.image_buffer = []
        self.bg_buffer = []
        for i in range(len(self.pathlist)):
            self.image_buffer.append(open(self.pathlist[i], "rb").read() )
        for i in range(len(self.pathbglist)):
            self.bg_buffer.append(open(self.pathbglist[i], "rb").read() )

        self.gen_model()
        
        
        
        
        
        
    def gen_model(self):
        
        model  = Sequential()
        
        
        model.add(Conv2D(self.filter_count, self.kernel_size, input_shape = self.input_shape ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
#        
        model.add(Conv2D(self.filter_count, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*2, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(Conv2D(self.filter_count*4, self.kernel_size ))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        
        model.add(Flatten())
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
#        model.add(Dropout(0.2))
        
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = self.leakrelu_alpha))
        
        
        
        model.add(Dense(8))
        model.add(Activation('linear'))
        
        
        op = Adam(lr=0.0001)
        model.compile(optimizer = op, loss = 'mse' )
        model.summary()
        self.model = model
        
    def gen_data(self,input_path,bg_path):
        
        
        self.pad_param = np.random.uniform(0,50)
        self.rotate_degree_param = np.random.choice([1,15])
        self.trim_param = np.random.uniform(0,15)
        self.brightness_param = 0.5
        self.contrast_param = 0.5
        
        
        
        bg_img = Image.open(bg_path)
        
        
        
        
        img = Image.open(input_path)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.normal(1,0.25))
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(np.random.normal(1,0.25))
        
        
#            img = img.resize((input_shape[0],input_shape[1]))
        img = np.asarray(img)
    #    img = img /255
    
        pad_top = int(abs(np.random.normal(0,self.pad_param)))
        pad_bottom = int(abs(np.random.normal(0,self.pad_param)))
        pad_left = int(abs(np.random.normal(0,self.pad_param)))
        pad_right = int(abs(np.random.normal(0,self.pad_param)))
        
        
        trim_top = int(abs(np.random.normal(0,self.trim_param)))
        trim_bottom = int(abs(np.random.normal(0,self.trim_param)))
        trim_left = int(abs(np.random.normal(0,self.trim_param)))
        trim_right = int(abs(np.random.normal(0,self.trim_param)))
     
        
        rotate_param = np.random.normal(0,self.rotate_degree_param)
        
        ret_img = Image.fromarray(img.astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
        ret_img = cv2.copyMakeBorder( np.asarray(ret_img), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        

        
        ret_img= cv2.resize(ret_img, dsize=(self.input_shape[0], self.input_shape[1]))
        
        
        ####compute mask
        mask_img = np.zeros((img.shape[0],img.shape[1],3))
        mask_2 = Image.fromarray(mask_img.astype('uint8')).rotate(rotate_param,resample = Image.NEAREST,expand = True, fillcolor = (255,255,255))
        mask_2 = cv2.copyMakeBorder( np.asarray(mask_2), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        
        bg_img = np.asarray(bg_img)
        ssss= bg_img.shape
        if(len(ssss)==2):
            bg_img = cv2.cvtColor(bg_img,cv2.COLOR_GRAY2RGB)
        bg_img = cv2.resize(bg_img, dsize=(self.input_shape[0], self.input_shape[1]))
        mask_3 = cv2.resize(mask_2, dsize=(self.input_shape[0], self.input_shape[1]))
        
        mask_3 = mask_3/255
        ret_img = ret_img*(1-mask_3) + bg_img*mask_3
#        ret_img[mask_3>=127]=bg_img[mask_3>=127]
        ret_img = (ret_img/127.5)-1
        
        x0 = trim_left
        x1 = ret_img.shape[0] - trim_right
        y0 = trim_top
        y1 = ret_img.shape[1] - trim_bottom
        sx = ret_img.shape[0] - trim_right - trim_left
        sy = ret_img.shape[1] - trim_top - trim_bottom
        ret_img = ret_img[y0:y1,x0:x1]
        ret_img= cv2.resize(ret_img, dsize=(self.input_shape[0], self.input_shape[1]))
        
        
        loc_topleft_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_topleft_img[0][0]=255
        
        loc_bottomleft_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_bottomleft_img[-1][0]=255
        
        loc_topright_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_topright_img[0][-1]=255
        
        loc_bottomright_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_bottomright_img[-1][-1]=255
        
        
        array_image = [loc_topleft_img,loc_bottomleft_img,loc_topright_img,loc_bottomright_img ]
        
        
        for i in range(len(array_image)):
                
                ## rotate 
            img2 = Image.fromarray(array_image[i].astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (0,0,0))
            
            # pad border
            img3 = cv2.copyMakeBorder( np.asarray(img2), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(0,0,0))
            
#            img3 = cv2.resize(img3, dsize=(self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_LANCZOS4)
    
            array_image[i] = img3
        
        
        
        
        ind = np.unravel_index(np.argmax(array_image[0], axis=None), array_image[1].shape)
        y_topleft = (ind[0] /array_image[0].shape[0]*self.input_shape[0] - trim_top)/sy*self.input_shape[0]
        x_topleft = (ind[1] /array_image[0].shape[1]*self.input_shape[1] - trim_left)/sx*self.input_shape[1]
        ind = np.unravel_index(np.argmax(array_image[1], axis=None), array_image[1].shape)
        y_bottomleft = (ind[0] /array_image[0].shape[0]*self.input_shape[0] - trim_top)/sy*self.input_shape[0]
        x_bottomleft = (ind[1] /array_image[0].shape[1]*self.input_shape[1]  - trim_left)/sx*self.input_shape[1]
        ind = np.unravel_index(np.argmax(array_image[2], axis=None), array_image[1].shape)
        y_topright = (ind[0] /array_image[0].shape[0]*self.input_shape[0] - trim_top)/sy*self.input_shape[0]
        x_topright = (ind[1] /array_image[0].shape[1]*self.input_shape[1] - trim_left)/sx*self.input_shape[1]
        ind = np.unravel_index(np.argmax(array_image[3], axis=None), array_image[1].shape)
        y_bottomright = (ind[0] /array_image[0].shape[0]*self.input_shape[0] - trim_top)/sy*self.input_shape[0]
        x_bottomright = (ind[1] /array_image[0].shape[1]*self.input_shape[1] - trim_left)/sx*self.input_shape[1]
        target = [x_topleft, y_topleft, x_topright, y_topright, x_bottomright, y_bottomright, x_bottomleft, y_bottomleft]
        return ret_img, target
    
    def plot_corner(self,img,target,real_target,color = (255,0,0)):
        ret_img = (img+1)*127.5
        ret_img = Image.fromarray(ret_img.astype('uint8'))
        shape = img.shape
        
        draw = ImageDraw.Draw(ret_img)
        draw.polygon([(target[0],target[1]),(target[2],target[3]),(target[4],target[5]),(target[6],target[7]) ],  outline=color)
        draw.polygon([(real_target[0],real_target[1]),(real_target[2],real_target[3]),(real_target[4],real_target[5]),(real_target[6],real_target[7]) ],  outline=(0,255,0))
        draw.ellipse((target[0]-5, target[1]-5, target[0]+5, target[1]+5), fill = 'blue', outline ='blue')
        draw.ellipse((target[4]-5, target[5]-5, target[4]+5, target[5]+5), fill = 'green', outline ='green')
        
        
        
#        draw.polygon([(target[0],target[1]),(target[2],target[3]),(target[4],target[5]),(target[6],target[7]) ],  outline=color)
        return ret_img
    
    def viz_result(self,folder_name, img_count):
        
        folder_path = os.path.join(save_path,'viz', folder_name )
        try:
            os.makedirs(folder_path)
        except:
            print('folder ' + folder_name +' already axist')
            
        permu = np.random.permutation(list(range(len(self.pathlist))))
        for i in range(img_count):
            img,target = self.gen_data(io.BytesIO(self.image_buffer[permu[i]]))
            indput_data = np.expand_dims(img, axis = 0)
            predict_corner = self.model.predict(indput_data)[0]
            viz_img = self.plot_corner(img,predict_corner,target)
            viz_img.save(os.path.join(folder_path, str(i) + '.jpg'))
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = len(self.pathlist) // batch_size
        permu_ind = list(range(len(self.pathlist)))
        
        
        
        for epoch in range(start_epoch,max_epoch):
            
            permu_ind = np.random.permutation(permu_ind)
            
            epoch_loss = []
        
            for step_index in range(max_step):
                batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                batch_target = np.zeros((batch_size,8))
                
                
                
                
                for batch_index in range(batch_size):
                    
                    img,target = self.gen_data(io.BytesIO(self.image_buffer[  step_index * batch_size +  batch_index ]),io.BytesIO(self.bg_buffer[  np.random.randint(0,len(self.bg_buffer)) ]))
                    batch_img[batch_index] = img
                    batch_target[batch_index] = target
                
                ##got batch
                #debug
                
                
                
                loss = self.model.train_on_batch(batch_img,batch_target)
                
#                sys.stdout.write(CURSOR_UP_ONE)
#                sys.stdout.write(ERASE_LINE)
                sys.stdout.write('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(loss))

                epoch_loss.append(loss)
                
                img_viz,target = self.gen_data(io.BytesIO(self.image_buffer[np.random.randint(0,len(self.pathlist))]),io.BytesIO(self.bg_buffer[  np.random.randint(0,len(self.bg_buffer)) ]))
                indput_data = np.expand_dims(img_viz, axis = 0)
                predict_corner = self.model.predict(indput_data)[0]
                viz_img = self.plot_corner(img_viz,predict_corner,target)
                try:
                    viz_img.save(os.path.join(save_path, 'debug' + '.jpg'))
                
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    
            print('average_loss = ' + str(np.average(epoch_loss))) 
            print(' ')
#            if(epoch % viz_interval ==0):
#                    self.viz_result('epoch_' + str(epoch), 6 )
#                    self.model.save_weights(os.path.join(save_path,'weight','epoch_' + str(epoch) + '.h5'))
                    
            self.model.save_weights('checkpoint_4corner_linear_small.h5')
        
            
            
            
        

if __name__ == '__main__':
    cor = corner()
    
    cor.model.load_weights(r'checkpoint_4corner_linear_small.h5')
#    cor.viz_result('test',6)
    cor.train(1,10000000,8,10)
    

        
        
        