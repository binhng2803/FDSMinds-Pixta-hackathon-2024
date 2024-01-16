import os
import cv2

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Lambda,
        Dense,
        BatchNormalization
    )
from keras import backend as K
from keras.optimizers import Adam


# ---------------------------------------


def baseModel() -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model


def loadModel(
) -> Model:
    custom_optimizer = Adam(learning_rate=0.005)
    model = baseModel()
    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)

    output = Activation("softmax")(base_model_output)
    # output = Dense(units=6, activation='softmax')(base_model_output)
    age_model = Model(inputs=model.input, outputs=output)
    age_model.load_weights("./age_model_weights.h5")
    age_model.compile(loss='categorical_crossentropy',
              optimizer=custom_optimizer,
              metrics=['accuracy'])
    return age_model


class Age_model ():
    def __init__(self):
        self.model = loadModel()
    
    def predict_age(self,image):
        img = cv2.imread(image)
        img = cv2.resize(img, (224,224))  # Đảm bảo kích thước ảnh phù hợp với mô hình
        img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        img = np.expand_dims(img, axis=0) 
        return self.__findApparentAge(self.model.predict(img))

    def __findApparentAge(self,age_predictions) -> np.float64:
        output_indexes = np.array(list(range(0, 101)))
        apparent_age = np.sum(age_predictions * output_indexes)
        if apparent_age <= 5 :
            return "Baby"
        if apparent_age <= 13 :
            return "Kid"
        if apparent_age <= 20 :
            return "Teenager"
        if apparent_age <= 40 :
            return "20-30S"
        if apparent_age <= 60 :
            return "40-50S"
        return "Senior"
    
a = Age_model()
print(a.predict_age("/mnt/d/AI_hakinthon/age/train/20-30s/px2054.jpg"))