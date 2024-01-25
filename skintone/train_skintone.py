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

    model.compile(
        # loss='categorical_crossentropy',
        loss="mse",
        optimizer="Adam",
        metrics=['accuracy'])
    return model

from keras.preprocessing.image import ImageDataGenerator
img_size = (64, 64)
batch_size = 256

datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.5, 1.5],
    rotation_range=20,
    horizontal_flip = True,
    validation_split=0.2
)
train_generator = datagen.flow_from_directory(
    "/mnt/d/AI_hakinthon/skintone/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # hoặc 'categorical' nếu bạn có nhiều lớp
    subset='training',  # chỉ định subset là 'training' cho dữ liệu huấn luyện
    seed=42
)
print(train_generator.class_indices)
# Tạo generator cho dữ liệu validation
validation_generator = datagen.flow_from_directory(
    "/mnt/d/AI_hakinthon/skintone/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # hoặc 'categorical' nếu bạn có nhiều lớp
    subset='validation',  # chỉ định subset là 'validation' cho dữ liệu kiểm định
    seed=42
)
checkpoint_callback = ModelCheckpoint(filepath='checkpoint_skintone/model_checkpoint_{epoch:02d}_{accuracy}.h5',
                                      save_freq='epoch',
                                      period=5)
model = loadModel()
model.load_weights('skintone_model.h5')
model.fit(x=train_generator,validation_data=validation_generator,validation_steps=512,epochs=50,callbacks=[checkpoint_callback])
model.save("skintone_model.h5")