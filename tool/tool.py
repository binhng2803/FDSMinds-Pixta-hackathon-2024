import ast
import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap,QImage
from PyQt5 import Qt
from PyQt5.uic import loadUi
# import module
import csv
import cv2

def add_multiple_captions_to_box(image, box, captions):
    image_with_captions = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(image_with_captions, (x1, y1), (x1+x2, y1+y2), (255, 255, 0), 2)
    line_height = 20
    caption_position = (x1, y1+y2 + 10)
    for caption in captions:
        cv2.putText(image_with_captions, caption, caption_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        caption_position = (caption_position[0], caption_position[1] + line_height) 
    return image_with_captions

class ImageTools():
    def __init__(self,csv_path,img_dir) -> None:
        self.data_list = []
        self.idx = 0
        self.img_dir = img_dir
        # print(self.img_dir)
        self.csv_path = csv_path
        with open(csv_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                self.data_list.append(row)
        
        
    def get_row(self,idx):
        row =self.data_list[idx]
        # print(row[0][3:])
        bbox =ast.literal_eval(row[1])
        try:
            image_path = f"{self.img_dir}/{row[0]}"
            captions = row[2:8]
        # print(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image_path = f"{self.img_dir}/{row[0][3:]}"
            captions = row[2:8]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        
        image_with_captions = add_multiple_captions_to_box(image,bbox, captions)
        # cv2.imshow("Image with Captions", image_with_captions)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.cvtColor(image_with_captions, cv2.COLOR_BGR2RGB)
        return image_with_captions,captions

    def next(self):
        if self.idx < len(self.data_list):
            self.idx += 1
            return self.get_row(self.idx)
        
    def back(self):
        if self.idx >0 :
            self.idx -= 1
            return self.get_row(self.idx)
        
    
    def update(self,g,r,a,m,s,e):
        # race,age,emotion,gender,skintone,masked
        self.data_list[self.idx][2] = r
        self.data_list[self.idx][3] = a
        self.data_list[self.idx][4] = e
        self.data_list[self.idx][5] = g
        self.data_list[self.idx][6] = s
        self.data_list[self.idx][7] = m
        self.save_to_csv()

    def delete(self):
        del self.data_list[self.idx]
        self.save_to_csv()

    def save_to_csv(self):
        with open(self.csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(self.data_list)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('test.ui', self)
        self.__init_ui()
        self.helper = ImageTools("labelcheck.csv","bbox/kaggle/working/bbox")
        self.btn_next.clicked.connect(self.__btn_next_clicked)
        self.btn_back.clicked.connect(self.__btn_back_clicked)
        self.btn_update.clicked.connect(self.__btn_update_clicked)
        self.btn_del.clicked.connect(self.__btn_del_clicked)
        self.__show_row()
        self.show()

    def __init_ui(self):
        self.cb_g.addItems(["Male","Female"])
        self.cb_r.addItems(["Negroid","Caucasian","Mongoloid"])
        self.cb_a.addItems(["Baby","Kid","Teenager","20-30s","40-50s","Senior"])
        self.cb_m.addItems(["unmasked","masked"])
        self.cb_s.addItems(["mid-light","light","mid-dark","dark"])
        self.cb_e.addItems(["Fear","Disgust","Surprise","Anger","Sadness","Neutral","Happiness"])

    def __btn_next_clicked(self):
        self.helper.next()
        self.__show_row()
    def __btn_back_clicked(self):
        self.helper.back()
        self.__show_row()
    def __btn_update_clicked(self):
        self.helper.update(self.cb_g.currentText(),self.cb_r.currentText(),self.cb_a.currentText(),self.cb_m.currentText(),self.cb_s.currentText(),self.cb_e.currentText())
        self.__show_row()
    def __btn_del_clicked(self):
        self.helper.delete()
        self.__show_row()

    def __show_row(self):
        img , labels = self.helper.get_row(self.helper.idx)
        self.cb_g.setCurrentText(labels[3])
        self.cb_r.setCurrentText(labels[0])
        self.cb_a.setCurrentText(labels[1])
        self.cb_m.setCurrentText(labels[5])
        self.cb_s.setCurrentText(labels[4])
        self.cb_e.setCurrentText(labels[2])
        height, width, channel = img.shape

        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap.scaledToWidth(1280)
        self.label_7.setPixmap(pixmap)
        self.label_7.setScaledContents(True)


    
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
