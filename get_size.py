import pandas as pd
# import shutil
import os
import cv2
import matplotlib.pyplot as plt

def getsize(csv_filename, root_path):
    path_lst = []
    # pre_path = "af2020cv-2020-05-09-v5-dev/data"


    data_file = pd.read_csv(csv_filename)
    id_tuple = tuple(data_file["FileID"].values.tolist())
    classes_tuple = tuple(data_file["SpeciesID"].values.tolist())

    for i in range(len(id_tuple)):
        new_path = os.path.join(root_path, str(classes_tuple[i]))
        img_path = os.path.join(new_path,id_tuple[i]+".jpg")
        # print(imgpath)
        # print(repr(img_path))
        img = cv2.imread(img_path)
        height = img.shape[0]
        width = img.shape[1]
        all_h.append(height)
        all_w.append(width)

def draw_pic(heights, widths):
    plt.rcParams['font.sans-serif'] = ['SimHei']				# 解决中文无法显示的问题
    plt.rcParams['axes.unicode_minus'] = False			# 用来正常显示负号
    plt.subplot(211)														# 2行1列第一幅图
    plt.hist(sorted(heights), density=False, bins=[224,232,250,280,340,418,492,564,600])								# bins表示分为5条直方，可以根据需求修改
    plt.xlabel('Image height')
    plt.ylabel('Frequency')
    plt.subplot(212)														# 2行2列第二幅图
    plt.hist(sorted(widths), density=False, bins=[224,232,250,280,340,418,492,564,600])					# density为True表示频率，否则是频数，可根据需求修改
    plt.xlabel('Image width')
    plt.ylabel('Frequency')
    plt.show()





train_root_path = "./train"  #待训练图片存储文件夹的路径
test_root_path = "./test"    #训练时验证集存储文件夹的路径
train_filename = 'af2020cv-2020-05-09-v5-dev/training.csv'   #待读取的训练csv文件
test_filename = 'af2020cv-2020-05-09-v5-dev/annotation.csv'  #待读取的验证csv文件


# 生成ImageFolder所要求的图片格式
if __name__ == '__main__':
    all_h = []
    all_w = []
    getsize(train_filename, train_root_path)
    getsize(test_filename, test_root_path)
    print("getting image size is finished!")
    print(len(all_h))
    draw_pic(all_h, all_w)
    print("drawing picture is finished!")