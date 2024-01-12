import numpy as np
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

class FaceDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
    
    def load_model(self):
        model = YOLO('./models/detect/yolov8n/latest/best.pt')
        model.fuse()
        return model
    
    def predict(self, img):
        results = self.model(img)
        return results
    
    def get_bboxes(self, img):
        results = self.predict(img)
        bboxes = results[0].boxes.data.cpu().numpy()/np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0], 1, 1])
        return bboxes
    
    def crop(self, img, box):
        H, W, _ = img.shape
        x1, y1, x2, y2 = box
        x1 = int(x1*W*0.95)
        x2 = min(int(x2*W*1.05), int(W)) 
        y1 = int(y1*H*0.95)
        y2 = min(int(y2*H*1.05), int(H))
        return img[y1:y2, x1:x2, :]
        
    def btc_bbox(self, img):
        bboxes = self.get_bboxes(img)
        bboxes = [list(box) for box in list(bboxes)]
        output = []
        for i in range(len(bboxes)):    
            x1 = int(bboxes[i][0]*img.shape[1]) 
            y1 = int(bboxes[i][1]*img.shape[0]) 
            x2 = int(bboxes[i][2]*img.shape[1])
            y2 = int(bboxes[i][3]*img.shape[0])
            cropped = self.crop(img, bboxes[i][:4])
            output.append({'bbox': [x1, y1, x2-x1, y2-y1],
                           'cropped':cropped}) 
        return output
        
    
    def plot_bboxes(self, img):
        bboxes = self.get_bboxes(img)
        bboxes = [list(box) for box in list(bboxes)]
        output = np.copy(img)
        for i in range(len(bboxes)):    
            output = cv2.rectangle(output, (int(bboxes[i][0]*img.shape[1]), int(bboxes[i][1]*img.shape[0])), (int(bboxes[i][2]*img.shape[1]), int(bboxes[i][3]*img.shape[0])), (0, 255, 0), 2)
            output = cv2.putText(output, f'text {i+1}', (int(bboxes[i][0]*img.shape[1]), int(bboxes[i][1]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return output
    
    def show_cropped_images(self, image):
        bboxes = self.get_bboxes(image)
        bboxes = [list(box) for box in list(bboxes)]
        print(bboxes)
        n = len(bboxes)
        fig, axes = plt.subplots()
        for i in range(n):
            img = self.crop(image, bboxes[i][:4])
            axes.imshow(img)
        plt.show()
        
if __name__ == '__main__':
    detector = FaceDetector()
    img = cv2.imread('./yolo_dataset/valid/images/1016585.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector.show_cropped_images(img)
    # img = detector.plot_bboxes(img)
    # plt.imshow(img)
    # plt.show()

