# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:44:41 2019

@author: Gireesh Sundaram
"""

import cv2
import numpy as np
import pandas as pd
import codecs
import math

from configuration import window_height, window_width, window_shift, MPoolLayers_H, nb_labels

from keras.preprocessing import sequence

#%%
#reading the class files
data = {}
with codecs.open("Data/class.txt", 'r', encoding='utf-8') as cF:
    data = cF.read().split('\n')
    
#%%
def returnClasses(string):
    text = list(string)
    text = ["<SPACE>"] + ["<SPACE>" if x==" " else x for x in text] + ["<SPACE>"]
    classes = [data.index(x) if x in data else 2 for x in text]
    classes = np.asarray(classes)
    return classes
    
infile = pd.read_csv("Data/list.csv")
#%%
def find_max_width(path):
    infile = pd.read_csv(path)
    max_width = 0
    for record in range(0, len(infile)):
        path = infile["Path"][record]
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        h, w = np.shape(image)
        
        if (h > window_height): factor = window_height/h
        else: factor = 1
        
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
        h, w = np.shape(image)
        
        if w / window_width < math.ceil(w / window_width):
            padding = np.full((window_height, math.ceil(w / window_width) * 64 - w), 255)
            image = np.append(image, padding, axis = 1)
        
        h, w = np.shape(image)
        if w > max_width: max_width = w
    return(max_width)
        
#%%
max_width = find_max_width("Data/list.csv")

def split_frames(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    h, w = np.shape(image)
    
    if (h > window_height): factor = window_height/h
    else: factor = 1
    
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h, w = np.shape(image)
    
    if w / window_width < math.ceil(w / window_width):
        padding = np.full((window_height, math.ceil(w / window_width) * 64 - w), 255)
        image = np.append(image, padding, axis = 1)
    
    h, w = np.shape(image)
    frames = np.full((max_width // window_width, window_height, window_width, 1), 255)
    
    for slide in range(0, w // window_width):
        this_frame = image[:, slide * window_width : (window_width) * (slide+1)]
        this_frame = np.expand_dims(this_frame, 2)
        frames[slide] = this_frame
        
    return frames

#%%
def prepareData(path):
    infile = pd.read_csv(path)

    x_train = np.zeros((len(infile), max_width // window_width, window_height, window_width, 1))    
    y_train = []
    im_train = []
    
    for record in range(0, len(infile)):
        print("Reading file: " + str (record))
        path = infile["Path"][record]
        annotation = infile["Annotation"][record]
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        h, w = np.shape(image)
        
        if (h > window_height): factor = window_height/h
        else: factor = 1
        
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC).T
        h, w = np.shape(image)
        
        im_train.append(image)
        y_train.append(returnClasses(annotation))
        
        x_train_len = np.asarray([len(im_train[i]) for i in range(len(im_train))])
        x_train_len = (x_train_len/18).astype(int)
        y_train_len = np.asarray([len(y_train[i]) for i in range(len(y_train))])
        
        x_train[record] = split_frames(path)
        
        y_train_pad = sequence.pad_sequences(y_train, value=float(nb_labels), dtype='float32', padding="post")

    return x_train, y_train_pad, x_train_len, y_train_len