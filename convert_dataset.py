import pandas as pd
import shutil
import os

def convert_dataset(csv_filename, pre_path, root_path):
    path_lst = []
    # pre_path = "af2020cv-2020-05-09-v5-dev/data"


    data_file = pd.read_csv(csv_filename)
    id_tuple = tuple(data_file["FileID"].values.tolist())
    classes_tuple = tuple(data_file["SpeciesID"].values.tolist())

    try:
        for i in range(len(id_tuple)):
            new_path = os.path.join(root_path, str(classes_tuple[i]))
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(os.path.join(pre_path, id_tuple[i]+".jpg"),os.path.join(new_path,id_tuple[i]+".jpg"))
    except:
        print("match error")


pre_path = "af2020cv-2020-05-09-v5-dev/data"  #图片文件之前所在文件夹
train_root_path = "images/train"  #待训练图片存储文件夹的路径
test_root_path = "images/test"    #训练时验证集存储文件夹的路径
train_filename = 'af2020cv-2020-05-09-v5-dev/training.csv'   #待读取的训练csv文件
test_filename = 'af2020cv-2020-05-09-v5-dev/annotation.csv'                             #待读取的验证csv文件


# 生成ImageFolder所要求的图片格式
if __name__ == '__main__':
    convert_dataset(train_filename, pre_path, train_root_path)
    convert_dataset(test_filename, pre_path, test_root_path)
    print("dataset converting is finished!")

