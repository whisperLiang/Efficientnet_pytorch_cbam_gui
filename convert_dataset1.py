import pandas as pd
import shutil
import os

def convert_dataset(file_dir, root_path):
    fid_list = []
    clsid_list = []
    # pre_path = "af2020cv-2020-05-09-v5-dev/data"
    for file in os.listdir(file_dir):
        fid_list.append(file)
        name = file.split(sep=' ')
        if 'g' in name[0]:
            clsid = int(name[0].replace('g', '')) - 1
            clsid_list.append(clsid)

    result = [[fid_list[i], clsid_list[i]] for i in range(len(clsid_list))]
    csv_pd = pd.DataFrame(columns=['FileID', 'SpeciesID'], data=result)
    csv_pd.to_csv('./train.csv')
    try:
        for i in range(len(fid_list)):
            new_path = os.path.join(root_path, str(clsid_list[i]))
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(os.path.join(file_dir, fid_list[i]),os.path.join(new_path,fid_list[i]))
    except:
        print("match error")


train_root_path = "images/train"  #待训练图片存储文件夹的路径
test_root_path = "images/test"    #训练时验证集存储文件夹的路径
train_dir1 = './data'
# train_dir2 = './data2'

# 生成ImageFolder所要求的图片格式
if __name__ == '__main__':
    convert_dataset(train_dir1, train_root_path)
    # convert_dataset(train_dir2, test_root_path)
    print("dataset converting is finished!")

