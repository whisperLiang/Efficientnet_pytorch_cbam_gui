# 需要的包
from __future__ import print_function, division

import json
import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import qtawesome
import torch
import torch.nn as nn
import torch.nn.functional as FUN
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from scipy import io
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import mymodel

# input_size = 224
class_num = 20
# image_dir = './test/15s/db00f57603d248ed2ddf86906d05ba85.jpg'
use_gpu = torch.cuda.is_available()

# %%

# 程序所需的链接和模型等
global selectModel
selectModel = "Efficientnet-b5"
# global Predict_Image_Path
# Predict_Image_Path = str()

# 读取预测的图像数据
# def readImage(imagePath):
#     #   添加文件图片数据
#     src = cv.imread(imagePath)
#     image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     image = cv.resize(
#         image,
#         (256, 256),
#         interpolation=cv.INTER_AREA
#     )
#     image = image.flatten()
#     return image

# 界面设计
class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()
        self.Predict_Image_Path = ""
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(1200, 800)
        self.setWindowTitle("基于多尺度融合的海洋生物分类系统")

        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        #  左侧设置
        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_label_1 = QtWidgets.QPushButton("本机测试")
        self.left_label_1.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.left_label_3.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "导入海洋生物图像")
        self.left_button_1.clicked.connect(self.loadFile)
        self.left_button_1.setObjectName('left_button')

        # self.left_button_2 = QtWidgets.QComboBox()
        # self.left_button_2.addItem('FACE-model')
        # self.left_button_2.addItem('BACK-model')
        # self.left_button_2.addItem('UP-model')
        # self.left_button_2.addItem('LEFT-model')
        # self.left_button_2.addItem('EYE-model')
        # self.left_button_2.currentIndexChanged.connect(self.selectionchange)

        self.left_button_3 = QtWidgets.QComboBox()
        self.left_button_3.addItem("Efficientnet-b5")
        self.left_button_3.addItem("Multi-scale fusion model")
        self.left_button_3.currentIndexChanged.connect(self.selectModel)

        #         self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.sellsy',color='white'),"选择部位")
        #         self.left_button_2.clicked.connect(self.selectSite)
        #         self.left_button_2.setObjectName('left_button')

        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.film', color='white'), "开始测试")
        self.left_button_4.setObjectName('left_button')
        self.left_button_4.clicked.connect(self.runTest)

        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.comment', color='white'), "反馈建议")
        self.left_button_5.setObjectName('left_button')

        self.left_button_6 = QtWidgets.QPushButton(qtawesome.icon('fa.star', color='white'), "关注我们")
        self.left_button_6.setObjectName('left_button')

        self.left_button_7 = QtWidgets.QPushButton(qtawesome.icon('fa.question', color='white'), "遇到问题")
        self.left_button_7.setObjectName('left_button')

        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_2, 3, 0,1,3)
        self.left_layout.addWidget(self.left_button_3, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 5, 0, 1, 3)

        self.left_layout.addWidget(self.left_label_3, 10, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 11, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_6, 12, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_7, 13, 0, 1, 3)

        # 右侧设置

        self.right_label_1 = QtWidgets.QLabel("显示海洋生物图像")
        self.right_label_1.setObjectName('right_label')
        self.right_layout.addWidget(self.right_label_1, 0, 0, 1, 10)
        self.right_label_1.setGeometry(QtCore.QRect(200, 200, 512, 512))
        self.right_label_1.setFrameShape(QtWidgets.QFrame.Box)
        self.right_label_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.right_label_1.setScaledContents(True)
        self.right_label_1.setStyleSheet('''
            #right_label{background-color: gray}
        ''')

        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        # self.left_button_2.setStyleSheet("""
        #   QComboBox{font-family:Microsoft YaHei;border:1px;
        #   border-color#252525:border-radius:2px;background: #404040;font:12px;color:white;height: 30px;}
        #   QComboBox:editable{background:black;}
        #   QComboBox QAbstractItemView{border: 0px;outline:0px;
        #   selection-background-color: blue;height:100px;background: rgb(1,58,80);color:white;font-size:12px}
        #   QComboBox QAbstractItemView::item {height:30px;}
        #   QComboBox QAbstractItemView::item:selected{background-color: blue;}
        #   QComboBox::down-arrow{image:url(application/resources/icons/combo_arrow.png);}
        #   QComboBox::drop-down{border:0px;}
        # """)
        self.left_button_3.setStyleSheet("""
          QComboBox{font-family:Microsoft YaHei;border:1px;
          border-color#252525:border-radius:2px;background: #404040;font:12px;color:white;height: 30px;}
          QComboBox:editable{background:black;}
          QComboBox QAbstractItemView{border: 0px;outline:0px;
          selection-background-color: blue;height:100px;background: rgb(1,58,80);color:white;font-size:12px}
          QComboBox QAbstractItemView::item {height:30px;}
          QComboBox QAbstractItemView::item:selected{background-color: blue;}
          QComboBox::down-arrow{image:url(application/resources/icons/combo_arrow.png);}
          QComboBox::drop-down{border:0px;}
        """)

        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        ''')

        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        # 整体设置
        self.setWindowOpacity(0.95)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.main_layout.setSpacing(0)

    #  加载图片
    def loadFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', r'./test',
                                               'Image files(*.jpg *.gif *.png)')
        self.right_label_1.setPixmap(QPixmap(fname))
        # image = self.right_label_1.text()
        self.Predict_Image_Path = str(fname)
        src = cv.imread(self.Predict_Image_Path)
        image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        image = cv.resize(
            image,
            (256, 256),
            interpolation=cv.INTER_AREA
        )
        image = image.flatten()
        image = np.reshape(image, (256, 256))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print(self.Predict_Image_Path)
        print("加载图像信息已完成")

    def selectModel(self, i):
        # 标签用来显示选中的文本
        # currentText()：返回选中选项的文本
        selectModel = self.left_button_3.currentText()
        print("选中的模型为" + selectModel)

    #  开始测试, 调用训练好的模型，导入图像信息进行预测
    def runTest(self):
        print("开始测试**********************")
        # selectStr = self.left_button_2.currentText()
        selectModel = self.left_button_3.currentText()
        print("识别海洋生物类别程序开始运行")
        # print(selectStr)
        print(selectModel)

        print("**********************")

        # ---------------------------------------
        # 实现分类
        # 初始化函数
        if selectModel == 'Efficientnet-b5':
            model = mymodel.Net_b5(num_class=class_num)
            if use_gpu:
                model = model.cuda()

            # GPU语句
            # model.load_state_dict(torch.load("./checkpoint/best_model_final.pth"))

            # CPU语句
            model.load_state_dict(torch.load("./checkpoint/best_model_final.pth", map_location='cpu'))
            model.eval()
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
            image = Image.open(self.Predict_Image_Path)
            # image = cv.imread(self.Predict_Image_Path)
            img = tfms(image).unsqueeze(0)
            if use_gpu:
                img = Variable(img.cuda())
            else:
                img = Variable(img)
            # print(img.shape) # torch.Size([1, 3, 224, 224])

            labels_map = json.load(open('./underwater.txt'))
            labels_map = [labels_map[str(i)] for i in range(20)]

            with torch.no_grad():
                outputs = model(img)
            # Print predictions
            print('-----')
            cout = 0
            for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
                cout += 1
                prob = torch.softmax(outputs, dim=1)[0, idx].item()
                # print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
                label = labels_map[idx]
                p = prob * 100

                print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
                QtWidgets.QMessageBox.information(self, "海洋生物识别结果及准确率", f"预测为 {label},准确率为{p}")


# 程序入口
def main():
    app = QtWidgets.QApplication(sys.argv)  # 实例化一个应用对象
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())  # 确保主循环安全退出


if __name__ == '__main__':
    main()

# %%
