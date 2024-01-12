from ultralytics import YOLO
import torch

yolo_org = YOLO('yolov8n.pt')

yolo_model = yolo_org.model.model
# print(yolo_org.model)
model_99 = yolo_model[-1]
bb_4 = yolo_model[:5]
bb_6 = yolo_model[5:7]
bb_9 = yolo_model[7:11]
print(bb_4)
print('___________________________________________')
print(bb_6)
print('________________________________________')
print(bb_9)
# print(backbone)
# print('___________________________________________')
neck = yolo_model[12]
# print(neck)

random_batch = torch.randn(8, 3, 224, 224)
print(random_batch.shape)
o4 =bb_4(random_batch)
print(o4.shape)

o6 =bb_6(o4)
print(o6.shape)
o9 =bb_9(o6)
print(o9.shape)
# output = model_99(random_batch)
# print(output.shape)
o96 = torch.cat((o9, o6), dim=1)
print(o96.shape)
n_output = neck(o96)
print(n_output.shape)