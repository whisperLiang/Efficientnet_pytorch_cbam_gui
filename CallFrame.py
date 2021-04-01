from __future__ import print_function, division
import sys
import time
# import serial  # 这个模块是通信模块
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QPalette
from SaveGesture import *
from test_one import test_model
from Frame import *
import torch
print(torch.__version__)
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as FUN
import os
from scipy import io
import json
from utils import mymodel
from PIL import Image, ImageDraw, ImageFont

global gesture_action
class_num = 10
use_gpu = torch.cuda.is_available()

class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)   # 在界面文件Frame中以及根据界面自动定义了
        self.initUI()

    def initUI(self):
        # 给按钮连接槽函数（CloseButton在Frame中自动连接了）
        self.GetGestureButton.clicked.connect(self.GetGesture)
        self.JudgeButton.clicked.connect(self.JudgeGesture)
        # self.ExcuteGestureButton.clicked.connect(self.ExcuteGesture)
        self.HelpButton.clicked.connect(self.Help)

        # 窗口设置美化
        self.setWindowTitle('手势识别')
        self.setWindowIcon(QIcon('./ges_ico/frame.ico'))
        self.resize(750, 485)

        # 线程操作用于显示时间
        self.initxianceng()

        # 单独给CloseButton添加标签
        self.CloseButton.setProperty('color', 'gray')  # 自定义标签
        self.GetGestureButton.setProperty('color', 'same')
        self.JudgeButton.setProperty('color', 'same')
        # self.ExcuteGestureButton.setProperty('color', 'same')
        self.HelpButton.setProperty('color', 'same')

    # 定义槽函数
    def GetGesture(self):
        self.LitResultlabel.setText("")
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/white.ico'))
        self.LitResultlabel.setAutoFillBackground(False)
        saveGesture()
        self.LitResultlabel.setText("已经将该图像保存在电脑本地")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)

    def JudgeGesture(self):
        global gesture_action  # 要修改全局变量需要先在函数里面声明一下
        self.LitResultlabel.setText("正在调用训练好的Effcientnet网络识别图像")
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        QApplication.processEvents()  # 这里需要刷新一下，否则上面的文字不显示
        model_ft = mymodel.Net_b5(num_class=class_num)
        # 离线加载预训练，需要事先下载好
        # model_ft = EfficientNet.from_name(net_name)
        # net_weight = 'eff_weights/' + pth_map[net_name]
        # state_dict = torch.load(net_weight)
        # model_ft.load_state_dict(state_dict)

        # 修改全连接层
        # num_ftrs = model_ft._fc.in_features
        # model_ft._fc = nn.Linear(num_ftrs, class_num)
        if use_gpu:
            model_ft = model_ft.cuda()

        # GPU语句
        # model_ft.load_state_dict(torch.load("./checkpoint/best_model_final.pth"))
        # CPU语句
        model_ft.load_state_dict(torch.load("./checkpoint_gesture/best_model_final.pth", map_location='cpu'))

        gesture_num = test_model(model_ft)
        self.result_show(gesture_num)

    def Help(self):
        QMessageBox.information(self, "操作提示框", "获取手势：通过OpenCV和摄像头获取一张即时照片。\n"
                                "判断手势：通过之前训练好的参数和卷积神经网络判断手势。\n"
                                "执行手势：根据识别的手势姿态控制机械手作业。")

    def result_show(self,gesture_num):
        # self.LitResultlabel.setText(f"判断结果：该手势数字为:{gesture_num[0]},其识别正确率为:{gesture_num[1]}")
        QMessageBox.information(self, "手势数字识别结果及准确率", f"识别结果为 {gesture_num[0]},准确率为{gesture_num[1]}")
        self.LitResultlabel.setAutoFillBackground(True)  # 允许上色

        palette = QPalette()                          # palette 调色板
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.LitResultlabel.setPalette(palette)
        self.ImaResultlabel.setToolTip('这是一个示意图片结果')  # 鼠标放在上面出现提示框
        self.ImaResultlabel.setPixmap(QPixmap('./ges_ico/ges1.ico'))
        self.LitResultlabel.setAlignment(Qt.AlignCenter)
        self.ImaResultlabel.setAlignment(Qt.AlignCenter)


    def initxianceng(self):
        # 创建线程
        self.backend = BackendThread()
        # 信号连接槽函数
        self.backend.update_date.connect(self.handleDisplay)
        # 开始线程
        self.backend.start()

    # 将当期时间输出到文本框
    def handleDisplay(self, data):
        self.statusBar().showMessage(data)


# 后台线程更新时间
class BackendThread(QThread):
    update_date = pyqtSignal(str)

    def run(self):
        while True:
            date = QDateTime.currentDateTime()
            currTime = date.toString('yyyy-MM-dd hh:mm:ss')
            self.update_date.emit(str(currTime))
            time.sleep(1)  # 推迟执行的1秒


if __name__ == "__main__":
    app = QApplication(sys.argv)  # sys.argv是一个命令行参数列表
    myWin = MyMainWindow()
    myWin.setObjectName('Window')
    # 给窗口背景上色
    qssStyle = '''
              QPushButton[color='gray']{
              background-color:rgb(205,197,191)
              }
              QPushButton[color='same']{
              background-color:rgb(225,238,238)
              }
              #Window{
              background-color:rgb(162,181,205) 
              }
              '''
    myWin.setStyleSheet(qssStyle)
    myWin.show()
    sys.exit(app.exec_())

