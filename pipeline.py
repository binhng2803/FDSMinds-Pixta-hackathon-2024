from detect_models import FaceDetector
from classify_models import create_classify_models, predict
import cv2

detector = FaceDetector()
models = create_classify_models()

def pipeline(path):
    img = cv2.imread(path)
    btc_bbox = detector.btc_bbox(img)
    for box in btc_bbox:
        cropped = box['cropped']
        gender, age, race, skintone, emotion, masked = predict(models, cropped)
        box['gender'] = gender
        box['age'] = age
        box['race'] = race
        box['skintone'] = skintone
        box['emotion'] = emotion
        box['masked'] = masked
    return btc_bbox

if __name__ == '__main__':
    path = './yolo_dataset/valid/images/105767349.jpg'
    btc_bbox = pipeline(path)
    print(btc_bbox)

    

    
