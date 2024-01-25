from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO('yolov8.yaml')
    yolo.train(data= "config.yaml", epochs=1, batch= 20, workers= 1, optimizer= 'Adam', imgsz= 320, lr0= 0.0001, task= 'detect', amp= False)