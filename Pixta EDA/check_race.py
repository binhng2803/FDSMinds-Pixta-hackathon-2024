import pandas as pd
import cv2
import os
import numpy as np
# import shutil
# from PIL import Image
PREFIX = "_race_final"
data=pd.read_csv("./check_mongoloid_final.csv")
# print(data.head(5)) 
# new_height=640
# new_width=640
folder_img=r"E:\Pixta Hackathon\dataset\mnt\md0\projects\sami-hackathon\private\data"
image_path=list(set(data["file_name"].values.tolist()))
index=0
n=len(image_path)
cols=data.columns.tolist()
caucasian=[]
mongoloid=[]
negroid=[]
skintone_false=[]
gender_false=[]
# other_false=[]
dim=1280
while index<n:
    img=cv2.imread(os.path.join(folder_img,image_path[index]),cv2.COLOR_RGB2BGR)
    height,width=img.shape[:-1]
    detail=data[data["file_name"]==image_path[index]]
    # print(image_path[index])
    # print(detail["bbox"])
    # print(detail[["height","width"]])
    # print(img_rec.shape)
    for i in range(len(detail)):
        bbox=detail.iloc[i,3]
        x,y,w,h=bbox.split(",")
        x,y,w,h=int(round(float(x.strip()[1:]),0)),int(round(float(y.strip()),0))\
            ,int(round(float(w.strip()),0)),int(round(float(h.strip()[:-1]),0))
        x1=int(x)
        y1=int(y)
        x2=int(x+w)
        y2=int(y+h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255),thickness=5)
        cv2.putText(img, f"race:{detail.iloc[i,5]}", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
    # if max(height,width) > dim:
    #     ratio=int(max(height,width)/dim)
    #     cv2.resize(img,(int(height/ratio),int(width/ratio)))
    cv2.namedWindow(image_path[index]+" ("+str(index+1)+")",cv2.WINDOW_NORMAL)
    cv2.imshow(image_path[index]+" ("+str(index+1)+")",img)
    cv2.resizeWindow(image_path[index]+" ("+str(index+1)+")",int(width/2),int(height/2))
    # cv2.resizeWindow(image_path[index],width,height)
    # cv2.waitKey(0)
    key=cv2.waitKey(0)
    if key==32:
        # cv2.destroyAllWindows()
        index+=1
        cv2.destroyAllWindows()
    elif key==27:
        cv2.destroyAllWindows()
        break
    elif key==49:
        caucasian+=detail.values.tolist()
        index+=1
        cv2.destroyAllWindows()
    elif key==50:
        mongoloid+=detail.values.tolist()
        index+=1
        cv2.destroyAllWindows()
    elif key==51:
        negroid+=detail.values.tolist()
        index+=1
        cv2.destroyAllWindows()
    elif key==52:
        skintone_false+=detail.values.tolist()
        index+=1
        cv2.destroyAllWindows()
    elif key==53:
        gender_false+=detail.values.tolist()
        index+=1
        cv2.destroyAllWindows()
    elif key==ord("z"):
        if index>0:
            index-=1
        elif index==0:
            index==0
        cv2.destroyAllWindows()
    # elif key==54:
    #     for j in range(len(detail)):
    #         other_false.iloc[index+j]=detail[j]
    #     index+=1
    #     cv2.destroyAllWindows()
    else:
        pass
pd.DataFrame(caucasian, columns=data.columns.tolist()).to_excel(f"caucasian{PREFIX}.xlsx",index=False)
pd.DataFrame(mongoloid, columns=data.columns.tolist()).to_excel(f"mongoloid{PREFIX}.xlsx",index=False)
pd.DataFrame(negroid, columns=data.columns.tolist()).to_excel(f"negroid{PREFIX}.xlsx",index=False)
pd.DataFrame(skintone_false, columns=data.columns.tolist()).to_excel(f"skintone_false{PREFIX}.xlsx",index=False)
pd.DataFrame(gender_false, columns=data.columns.tolist()).to_excel(f"gender_false{PREFIX}.xlsx",index=False)
# other_false.to_excel("other_false.xlsx",index=False)
   