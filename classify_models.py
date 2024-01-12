from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import cv2
from transform import test_transform
from config import classify_lst

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
    
def create_classify_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resgen = MyResnet(n_classes=2).to(device)
    resage = MyResnet(n_classes=6).to(device)
    resrace = MyResnet(n_classes=3).to(device)
    reskin = MyResnet(n_classes=4).to(device)
    resemo = MyResnet(n_classes=7).to(device)
    resmask = MyResnet(n_classes=2).to(device)
    
    resgen.load_state_dict(torch.load('./models/classify/classify_model/gender/best.pt', map_location=torch.device('cpu')))
    resage.load_state_dict(torch.load('./models/classify/classify_model/age/best.pt', map_location=torch.device('cpu')))
    resrace.load_state_dict(torch.load('./models/classify/classify_model/race/best.pt', map_location=torch.device('cpu')))
    reskin.load_state_dict(torch.load('./models/classify/classify_model/skintone/best.pt', map_location=torch.device('cpu')))
    resemo.load_state_dict(torch.load('./models/classify/classify_model/emotion/best.pt', map_location=torch.device('cpu')))
    resmask.load_state_dict(torch.load('./models/classify/classify_model/masked/best.pt', map_location=torch.device('cpu')))
    
    resgen.eval()
    resage.eval()
    resrace.eval()
    reskin.eval()
    resemo.eval()
    resmask.eval()
    
    return resgen, resage, resrace, reskin, resemo, resmask

def predict(models, img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = test_transform(img)
    img = img.unsqueeze(0)
    output = []
    for i in range(6):
        prediction = models[i](img)
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.item()
        output.append(classify_lst[i][prediction])
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