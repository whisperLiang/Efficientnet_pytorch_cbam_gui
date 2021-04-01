# 海洋生物分类
## 

# 代码说明

1. pip-requirements.txt 需要安装的库
2. convert_dataset.py 整理csv文件格式的数据集
3. creat_map.py 生成对应的标签映射
4. train.py 训练主函数
5. test_one.py 利用训练好的模型预测一张图片
6. test_all.py 预测整个test文件里的图片
7. test_tta. py 预测时加入tta，但是实际效果不好，不知道哪里出了问题
8. sys_gui .py 运行时生成界面，可实现单张图片的读取，以及对单张图片的预测

## 

# 训练方案

模型方面采用的是efficientnet-b5，在原始b5模型中增加了cbam注意力模块，数据增强方面使用了随机裁切、翻转、auto_augment、随机擦除以及cutmix, 损失函数采用CrossEntropyLabelSmooth，训练策略方面采用了快照集成（snapshot）思想。

第一阶段训练，图像输入尺寸为465，使用LabelSmooth和cutmix，采用带学习率自动重启的CosineAnnealingWarmRestarts方法，获得5个模型快照，选择val_acc最高的模型，作为第一阶段的训练结果。

运行指令为 !python train.py

第二阶段训练，图像输入尺寸为465，适当调整随机裁切和随机擦除的参数，增加weight_decay，在第一阶段模型的基础上训练获得5个模型快照，选择val_acc最高的模型，作为第二阶段的训练结果。

运行指令为 !python train.py --batch_size=10 --lr=5e-5 --image_size=456\

 --weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 \

 --epochs=100 --num_class=20 --model_path='checkpoint/best_model_456.pth'

第三阶段训练，图像输入尺寸为465，关闭cutmix，损失函数采用CrossEntropyLoss，在第二阶段模型的基础上训练获得5个模型快照，选择val_acc最高的模型，作为最终的训练结果。

运行指令为 !python train.py --batch_size=10 --lr=1e-6 --image_size=456\

 --weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 --cutmix\

 --label_smooth --epochs=100 --num_class=20 --model_path='checkpoint/best_model_456.pth'
