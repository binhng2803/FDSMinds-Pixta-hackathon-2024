from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch
import torch.nn as nn
import cv2
from transform import test_transform
from config import classify_lst
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

class MyResnet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.backbone = resnet50() #weights=ResNet50_Weights.DEFAULT)
        del self.backbone.fc
        self.fc = nn.Linear(2048, n_classes)
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
class MyEffnet(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.backbone.classifier[1] = nn.Linear(1280, n_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        return x
# ================================================================================================
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
        self.model.load_weights('./skintone/skintone_model.h5')
    
    def predict_skintone(self,img):
        # img = cv2.imread(image)
        img = cv2.resize(img, (64,64))  # Đảm bảo kích thước ảnh phù hợp với mô hình
        img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        img = np.expand_dims(img, axis=0) 
        return self.__findApparentSkintone(self.model.predict(img))

    def __findApparentSkintone(self,predictions) -> np.float64:
        classes ={'dark': 0, 'light': 1, 'mid-dark': 2, 'mid-light': 3}
        predicted_class_id = np.argmax(predictions[0])
        return [k for k, v in classes.items() if v == predicted_class_id][0]

# =================================================================================================
    
def create_classify_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resgen = MyEffnet(n_classes=2).to(device)
    resage = MyResnet(n_classes=6).to(device)
    resrace = MyEffnet(n_classes=3).to(device)
    reskin = MyEffnet(n_classes=4).to(device)
    resemo = MyEffnet(n_classes=7).to(device)
    resmask = MyResnet(n_classes=2).to(device)
    
    resgen.load_state_dict(torch.load('./models/classify/efficient/gender/best.pt', map_location=torch.device('cpu')))
    resage.load_state_dict(torch.load('./models/classify/classify_model/age/best.pt', map_location=torch.device('cpu')))
    resrace.load_state_dict(torch.load('./models/classify/efficient/race/best.pt', map_location=torch.device('cpu')))
    reskin.load_state_dict(torch.load('./models/classify/efficient/skintone/best.pt', map_location=torch.device('cpu')))
    resemo.load_state_dict(torch.load('./models/classify/efficient/emotion/best.pt', map_location=torch.device('cpu')))
    resmask.load_state_dict(torch.load('./models/classify/classify_model/masked/best.pt', map_location=torch.device('cpu')))
    
    resgen.eval()
    resage.eval()
    resrace.eval()
    reskin.eval()
    resemo.eval()
    resmask.eval()
    # reskin = Skintone_model()
    
    return resgen, resage, resrace, reskin, resemo, resmask

def predict(models, img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    skin_img = img
    skin_img = cv2.cvtColor(skin_img, cv2.COLOR_RGB2BGR)
    img = test_transform(img)
    img = img.unsqueeze(0)
    output = []
    for i in range(6):
        try:
            prediction = models[i](img)
            prediction = torch.argmax(prediction, dim=1)
            prediction = prediction.item()
            output.append(classify_lst[i][prediction])
        except:
            prediction = models[i].predict_skintone(skin_img)
            output.append(prediction)
    return output
    
if __name__ == "__main__":
    # resgen, resage, resrace, reskin, resemo, resmask = create_classify_models()
    # gender_list = ["female", "male"]
    # age_list = ["20-30s", "40-50s", "baby", "kid", "senior", "teenager"]
    # race_list = ["caucasian", "mongoloid", "negroid"]
    # skintone_list = ["dark", "light", "mid-dark", "mid-light"]
    # masked_list = ["masked", "unmasked"]
    # emotion_list = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    models = create_classify_models()
    
    img = cv2.imread('./cropped_face/race/valid/mongoloid/px548.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = predict(models, img)
    # img = test_transform(img)
    # img = img.unsqueeze(0)
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print(model)
    # random_batch = torch.randn(8, 3, 224, 224)
    # print(random_batch.shape)
    # output = []
    # output.append(resgen(img))
    # output.append(resage(random_batch).shape)
    # output.append(resrace(random_batch).shape)
    # output.append(reskin(random_batch).shape)
    # output.append(resemo(random_batch).shape)
    # output.append(resmask(random_batch).shape)
    print(output)