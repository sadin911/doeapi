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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    
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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

path_input = r'/home/chonlatid/py_project/DocData/trainset/**/'
path_test = r'/home/chonlatid/py_project/DocData/testset/**/'
bg_path = r'/home/chonlatid/py_project/docsegmentation/4connerwithsegment/images/background/**/'
save_path = r'/home/chonlatid/py_project/docsegmentation/4connerwithsegment'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif') # the tuple of file types

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'

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
        
        self.pathlist = []
        self.testlist = []
        self.pathbglist = []

        #self.pathbglist = glob2.glob(bg_path)

        for files in types:
            self.pathlist.extend(glob2.glob(join(path_input, files)))
        
        for files in types:
            self.testlist.extend(glob2.glob(join(path_test, files)))

        for files in types:
            self.pathbglist.extend(glob2.glob(join(bg_path, files)))

    def soft_dice_loss(self,y_true, y_pred, epsilon=1e-6): 
        ''' 
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
      
        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
            epsilon: Used for numerical stability to avoid divide by zero errors
        
        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
            
            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        '''
        
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * K.sum(y_pred * y_true, axes)
        denominator = K.sum(np.square(y_pred) + K.square(y_true), axes)
        
        return 1 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch
    
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
        # f = Embedding(2048,output_dim = 128)(f)
        # f = LSTM(512,return_sequences=True)(f)
        # f = GlobalMaxPool1D()(f)
        dn = Dense(1024)(f)
        dn = LeakyReLU(alpha=0.2)(dn)
        dn = Dense(8)(dn)
        outcorner = Activation('linear')(dn)
        
        
        model = Model(
			inputs=d0,
			outputs=[output_img, outcorner])
        
        return model
        
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        max_step = len(self.pathlist) // batch_size
        permu_ind = list(range(len(self.pathlist)))
        iteration = 0
        random.shuffle(self.pathlist)
        for epoch in range(start_epoch,max_epoch):
            permu_ind = np.random.permutation(permu_ind)
            epoch_loss = []
        
            for step_index in range(max_step):
                batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                batch_target = np.zeros((batch_size,8))
                batch_mask = np.zeros((batch_size,self.input_shape[0],self.input_shape[1]))
                
                for batch_index in range(batch_size//2):
                    img,target,mask = self.gen_data.gen_data((self.pathlist[  step_index * batch_size +  batch_index ]),(self.pathbglist[  np.random.randint(0,len(self.pathbglist)) ]))
                    batch_img[batch_index] = img
                    batch_target[batch_index] = target
                    batch_mask[batch_index] = mask
                
                for batch_index in range(batch_size//2,batch_size):
                    img,target,mask = self.gen_data.gen_data((self.pathlist[  step_index * batch_size +  batch_index ]),(self.pathbglist[  np.random.randint(0,len(self.pathbglist)) ]),isPerspective=True)
                    batch_img[batch_index] = img
                    batch_target[batch_index] = target
                    batch_mask[batch_index] = mask
                
                loss = self.model.train_on_batch(batch_img,[batch_mask,batch_target])
                train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                binary_loss = tf.keras.metrics.Mean('binary_loss', dtype=tf.float32)
                mse_loss = tf.keras.metrics.Mean('mse_loss', dtype=tf.float32)
                total_loss = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
                binary_loss(loss[1])
                mse_loss(loss[2])
                total_loss(loss[0])

                with train_summary_writer.as_default():
                    tf.summary.scalar('binary_loss', binary_loss.result() , step=iteration)
                    tf.summary.scalar('mse_loss', mse_loss.result() , step=iteration)
                    tf.summary.scalar('total_loss', total_loss.result() , step=iteration)
                
                # Reset metrics every epoch
                binary_loss.reset_states()
                mse_loss.reset_states()
                total_loss.reset_states()
                iteration += 1 
                        
                sys.stdout.write('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(loss))

                # epoch_loss.append(loss)
                
                if(step_index % viz_interval == 0): 
                    img_viz,target,mask = self.gen_data.gen_data((self.testlist[np.random.randint(0,len(self.testlist))]),(self.pathbglist[  np.random.randint(0,len(self.pathbglist))]),isPerspective=True)
                    indput_data = np.expand_dims(img_viz, axis = 0)
                    [predict_mask,predict_corner] = self.model.predict(indput_data)
                    self.model.save_weights('checkpoint_lr5_100_5_m250_r8_bg.h5')
                    viz_img = self.gen_data.plot_corner(img_viz,predict_corner[0])
                    gt_img= self.gen_data.plot_corner(img_viz,target)
                    predict_seg = self.gen_data.plot_corner_seg(predict_mask[0],target)
                    try:
                        predict_seg.save(os.path.join(save_path, 'mask' + '.jpg'))
                        viz_img.save(os.path.join(save_path, 'debug' + '.jpg'))
                        gt_img.save(os.path.join(save_path, 'gt' + '.jpg'))
                
                    except IOError as e:
                        print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    
            # print(' average_loss = ' + str(np.average(epoch_loss))) 
            #print(' ')

    def test(self,pathinput):
        img_viz = Image.open(pathinput)
        img_viz = img_viz.resize((256,256))
        img_viz = np.asarray(img_viz)
        img_viz = img_viz/127.5 - 1
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)

        [predict_mask,predict_corner] = self.model.predict(indput_data)
        viz_img = self.gen_data.plot_corner(img_viz,predict_corner[0])
        predict_seg = self.gen_data.plot_corner_seg(predict_mask[0],predict_corner[0])
        try:
            predict_seg.save(os.path.join(save_path, 'mask' + '.jpg'))
            viz_img.save(os.path.join(save_path, 'debug' + '.jpg'))

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    def warping(self,b64_input_image):
        byteImage = b64decode(b64_input_image)
        pilimg = Image.open(io.BytesIO(byteImage))
        pilimg = pilimg.resize((256,256))
        npimg = np.asarray(pilimg)
        npimg = npimg/127.2 - 1
        indput_data = img_viz
        indput_data = np.expand_dims(indput_data, axis = 0)

        [predict_mask,predict_corner] = self.model.predict(indput_data)
        viz_img = self.gen_data.plot_corner(img_viz,predict_corner[0])
        predict_seg = self.gen_data.plot_corner_seg(predict_mask[0],predict_corner[0])
    
if __name__ == '__main__':
    if(os.path.isdir('logs')):
        shutil.rmtree('logs')
    doc = DocScanner()
    doc.model.load_weights(r'checkpoint_lr5_100_5_m250_r8_bg_revert.h5')
    doc.test('images/input/20200123_091611.jpg')
    # gan.train(epochs=1000000, batch_size=1, sample_interval=100)