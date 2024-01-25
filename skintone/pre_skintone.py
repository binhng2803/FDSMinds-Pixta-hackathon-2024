import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
        BatchNormalization
    )
def loadModel(
) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))

    # model.compile(
    #     # loss='categorical_crossentropy',
    #     loss="mse",
    #     optimizer="Adam",
    #     metrics=['accuracy'])
    return model


class Skintone_model ():
    def __init__(self):
        self.model = loadModel()
        self.model.load_weights('skintone_model.h5')
    
    def predict_skintone(self,image):
        img = cv2.imread(image)
        img = cv2.resize(img, (64,64))  # Đảm bảo kích thước ảnh phù hợp với mô hình
        img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        img = np.expand_dims(img, axis=0) 
        return self.__findApparentSkintone(self.model.predict(img))

    def __findApparentSkintone(self,predictions) -> np.float64:
        classes ={'dark': 0, 'light': 1, 'mid-dark': 2, 'mid-light': 3}
        predicted_class_id = np.argmax(predictions[0])
        return [k for k, v in classes.items() if v == predicted_class_id][0]


a = Skintone_model()
res = a.predict_skintone("../cropped_face/skintone/valid/dark/px4.jpg")
print(res)