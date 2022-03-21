import streamlit as st
import cv2 as cv
import tempfile

import numpy as np
import cv2
import pandas as pd

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import BatchNormalization
from keras import Model
from keras.preprocessing import image
from keras.models import load_model

from tensorflow import keras
import tensorflow as tf

prediction_model = load_model('dashboard/prediction_model.h5')

# define feature extraction model

inputs = Input(shape=[224, 224, 1])

cnnModel = Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 1])(inputs)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Convolution2D(filters=64, kernel_size=3, activation='relu')(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Convolution2D(filters=128, kernel_size=3, activation='relu')(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Convolution2D(filters=256, kernel_size=3, activation='relu')(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Convolution2D(filters=512, kernel_size=3, activation='relu')(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Convolution2D(filters=1024, kernel_size=3, activation='relu')(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D(pool_size=2, strides=2)(cnnModel)

cnnModel = Dropout(0.5)(cnnModel)
outputs = Flatten()(cnnModel)

cnnModel = Model(inputs=inputs, outputs=outputs)
cnnModel.summary()

# Method to convert list of word to integers
def word_to_num_dictionary(list_of_list, key_col=0, val_col=1):
    value_dict = {}
    for i,value in enumerate(list_of_list):
      if i != 0:
        v = {value: str(i)}
        value_dict.update(v)
      else:
        v = {value: "null"}
        value_dict.update(v)
    return value_dict
    
# Method to convert integers to words
def num_to_word_dictionary(list_of_list, key_col=0, val_col=1):
    value_dict = {}
    for i,value in enumerate(list_of_list):
      if i != 0:
        v = {str(i):value}
        value_dict.update(v)
    return value_dict

# define word dictionary
words = ['null', 'repeat', 'help', 'he', 'hi', 'do', 'is', 'happy', 'you', 'my', 'very', 'appreciate', 'how', 'not', 'friend', 'she', 'i', 'can', 'please', 'congratulations', 'that', 'free', 'something', 'am', 'really', 'are', 'today', 'hiding', 'it', 'worry']
max_length = 30
word_to_num_dict =  word_to_num_dictionary(words)
num_to_word_dict =  num_to_word_dictionary(words)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False)[0][0][
        :, :max_length
    ]

    label = num_to_word(results[0])
    return label

def num_to_word(list_of_string):
  label=[]
  for k in list_of_string:
    if str(int(k)) in num_to_word_dict :
        label.append(num_to_word_dict[str(int(k))])
  return label


st.write("## Continues sign language recognition")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    vf = cv.VideoCapture(tfile.name)
    stframe = st.empty()
    
    frames_feature = []
    i = 0
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        i += 1
        if i % 2 == 0:
            continue

        img = cv2.resize(frame, (224, 224))

        # skin detection 
        # https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py

        # converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result = cv2.bitwise_not(global_mask)
                
        stframe.image(global_result)

        global_result = image.img_to_array(global_result)
        global_result = np.expand_dims(global_result, axis=0)
        r1 = cnnModel.predict(global_result)
        if len(frames_feature) <= 62: 
         frames_feature.append(r1[0])

    # pad missing frames
    for i in range(len(frames_feature), 63):
        paddingFeauters=[]
        for j in range(0,len(frames_feature[0])):
            arr1 = np.array([0])
            arr2 = np.array(paddingFeauters)
            paddingFeauters = np.concatenate((arr1, arr2))
        frames_feature.append(paddingFeauters)

    frames_feature = np.array(frames_feature)
    frames_feature = tf.ragged.constant([frames_feature])
    frames_feature = frames_feature.to_tensor()
    preds = prediction_model.predict(frames_feature)
    pred_texts = decode_batch_predictions(preds)
    st.info("predict output: "+str(pred_texts))