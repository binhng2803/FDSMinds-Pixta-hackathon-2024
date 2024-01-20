from ultralytics import YOLO

if __name__ == '__main__':

    yolo = YOLO('pretrained\yolov8s\\pbest.pt').cuda()
    yolo.train(data= "config.yaml", epochs=5, batch= 20, workers= 1, optimizer= 'Adam', imgsz= 320, lr0= 0.000001, task= 'detect', amp= False)
    # yolo = YOLO('yolov8n-cls.yaml')
    # yolo.train(data= "D:\Pixta\data_classification", epochs=11, batch= 32, workers= 5, optimizer= 'Adam', imgsz= 224, lr0= 0.001, task= 'classify')