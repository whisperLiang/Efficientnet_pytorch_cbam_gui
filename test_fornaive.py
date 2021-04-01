from __future__ import print_function, division

import torch
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

# input_size = 224
class_num = 20
use_gpu = torch.cuda.is_available()


def test_model(model, image_dir):
    model.eval()
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    image = Image.open(image_dir)
    img = tfms(image).unsqueeze(0)
    img = Variable(img.cuda())
    # print(img.shape) # torch.Size([1, 3, 224, 224])

    labels_map = json.load(open('./underwater.txt'))
    labels_map = [labels_map[str(i)] for i in range(20)]

    with torch.no_grad():
        outputs = model(img)
    # Print predictions
    print('-----')
    cout = 0
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        cout += 1
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))


if __name__ == '__main__':
    # answer_file = []
    # answer_pred = []
    # answer_file_name = 'outputs.csv'
    # os.makedirs('/cache/train', exist_ok=True)
    # os.makedirs('/cache/test', exist_ok=True)
    # data_root_path = '/cache/'
    # img_path = "/cache/data/"  # 图片文件之前所在文件夹
    # test_root_path = "/cache/test"  # 训练时验证集存储文件夹的路径
    # test_filename = os.path.join(data_root_path, 'test.csv')  # 待读取的验证csv文件

    # data_reference = get_data_reference(dataset="DatasetService", dataset_entity="image_dataset")  # 获取数据的OBS来源
    # for file_paths in data_reference.get_files_paths():  # 移动数据集文件到WebIDE中的空间里
    #     mox.file.copy(file_paths, os.path.join(
    #         '/cache/', file_paths.split('/')[-1]))
    # zip_file = zipfile.ZipFile('/cache/data.zip')  # 开始解压数据集
    # zip_list = zip_file.namelist()
    # for file in zip_list:
    #     zip_file.extract(file, '/cache/')

    val_data = pd.read_csv('./af2020cv-2020-05-09-v5-dev/annotation.csv')
    val_data["FileID"] = "test/" + val_data["SpeciesID"].astype(str) + "/" + val_data["FileID"] + '.jpg'
    # 自动下载到本地预训练
    # model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    # 移动预训练模型
    model_reference = get_data_reference(dataset="DatasetService", dataset_entity="model")  # 获取模型的OBS来源
    pretrained_model_path = model_reference.get_files_paths() # 获取模型的OBS地址
    print(pretrained_model_path)
    mox.file.copy(pretrained_model_path[0], os.path.join('pretrained_dir', pretrained_model_path[0].split('/')[-1])) # 移动模型到WebIDE中的存储空间里
    mox.file.copy(pretrained_model_path[1], './best_model_final.pth')
    print('[INFO] moving pretrained model successfully')
    # 移动训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_model_path = os.path.join(Context.get_project_path(), 'Debug', 'algo-run', 'model', 'best_model_456.pth')
    mox.file.copy(obs_model_path, model_path)
    model_ft = mymodel.Net_b5(num_class=classes)
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
    print('-' * 10)
    print('Test Accuracy:')
    model_ft.load_state_dict(torch.load(model_path))
    # criterion = nn.CrossEntropyLoss().cuda()
    length = len(test_csv["FileID"])
    for i in range(length):
        img = test_csv["FileID"].iat[i, 0]
        test_model(model_ft, img)