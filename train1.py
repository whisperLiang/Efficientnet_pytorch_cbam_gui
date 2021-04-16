import os, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import time, datetime
from PIL import Image, ImageOps, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import PIL

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from utils import loss, auto_augment, dataset, mymodel, tools, args
from MyEfficientNet import cbam_EfficientNet
import torch_optimizer as newoptim

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import datetime
print("1")

def validate(val_loader, model, criterion):
    batch_time = tools.AverageMeter('Time', ':6.3f')
    losses = tools.AverageMeter('Loss', ':.4e')
    top1 = tools.AverageMeter('Acc@1', ':2.2f')
    top5 = tools.AverageMeter('Acc@5', ':2.2f')
    progress = tools.ProgressMeter(len(val_loader), batch_time, losses)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = tools.accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # with summary_writer.as_default():
            #     tf.summary.scalar('val_loss', loss, step=i)
            #     tf.summary.scalar('val_acc@1', float(acc1), step=i)
            #     tf.summary.scalar('val_acc@5', float(acc5), step=i)
            # loss.cuda()
            # acc1.cuda()
            # acc5.cuda()
        return top1.avg.data.cpu().numpy(), losses.avg


# "创建监控对象"
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = 'log/' + current_time

# writer = SummaryWriter(log_dir)

def train(train_loader, model, criterion, optimizer, epoch, scheduler, mixup=False):
    batch_time = tools.AverageMeter('Time', ':6.3f')
    losses = tools.AverageMeter('Loss', ':.4e')
    top1 = tools.AverageMeter('Acc@1', ':2.2f')
    top5 = tools.AverageMeter('Acc@5', ':2.2f')
    progress = tools.ProgressMeter(len(train_loader), batch_time, losses,top1,top5)
    model.train()

    end = time.time()


    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        # print(input)
        target = target.cuda(non_blocking=True)
        # print(target)
        
        scheduler.step(epoch + i / len(train_loader))
        
        optimizer.zero_grad()
        if mixup == False:
            output = model(input)
            loss = criterion(output, target)
        else:    
            output, loss = tools.cutmix(input, target, model, criterion)

        acc1, acc5 = tools.accuracy(output, target, topk=(1, 5))
        # "写入监控数据"
        # writer.add_scalar("loss",loss,i)
        # writer.add_scalar("acc1",acc1,i)
        # writer.add_scalar("acc2",acc5,i)
        # writer.flush()
        # with summary_writer.as_default():
        #     tf.summary.scalar("loss",float(loss),step=i)
        #     tf.summary.scalar("acc1",float(acc1),step=i)
        #     tf.summary.scalar("acc2",float(acc5),step=i)
        # summary_writer.flush()
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 100 == 0) and (not i == 0):
            # with summary_writer.as_default():
            #     tf.summary.scalar('trian_loss', loss, step=i)
            #     tf.summary.scalar('train_acc@1', float(acc1), step=i)
            #     tf.summary.scalar('train_acc@5', float(acc5), step=i)
            # loss.cuda()
            # acc1.cuda()
            # acc5.cuda()
            progress.pr2int(i)
    return top1.avg.data.cpu().numpy(), losses.avg    

#保存val_acc最高的模型，acc相同则保存val_loss较低的模型    
def save_checkpoint(metrics, val_loss, val_acc, model, current_snapshot, fold_idx):
    if val_acc > metrics['best_acc']:
        metrics['best_acc'] = val_acc
        metrics['best_acc_loss'] = val_loss
        torch.save(model.state_dict(), './checkpoint/fold%d_snap%d.pth' % (fold_idx, current_snapshot))
        # torch.save(model.state_dict(), './checkpoint_ranger/fold%d_snap%d.pth' % (fold_idx, current_snapshot))
    elif val_acc == metrics['best_acc'] and val_loss < metrics['best_acc_loss']:
        metrics['best_acc'] = val_acc
        metrics['best_acc_loss'] = val_loss
        torch.save(model.state_dict(), './checkpoint/fold%d_snap%d.pth' % (fold_idx, current_snapshot))
        # torch.save(model.state_dict(), './checkpoint_ranger/fold%d_snap%d.pth' % (fold_idx, current_snapshot))

    if val_loss <= metrics['best_loss']:
        metrics['best_loss'] = val_loss
 
 #在不同的snapshot起始，重置metrics       

def reset_metrics(metrics, epoch, epochs, current_snapshot, snap_num):
    if epoch == epochs * (current_snapshot + 1) // snap_num:
        metrics['best_acc'] = 10
        metrics['best_acc_loss'] = 100
        metrics['best_loss'] = 100
        current_snapshot += 1
    return current_snapshot, metrics

train_df = pd.read_csv('./train.csv')
# print(f"train_df_pre:{train_df}")
train_df["FileID"] = "./images/train/" + train_df["SpeciesID"].astype(str)+ "/" + train_df["FileID"]

#分层划分为10折，可以进行十折交叉验证，但是比较耗时
skf = StratifiedKFold(n_splits=10, random_state=233, shuffle=True)
# print(f"skf is {list(skf.split(train_df['FileID'], train_df['SpeciesID']))}")


#训练时的可调超参数
train_batch_size = args.train_args.batch_size #训练时的batchsize
print(f"train_batch_size is {train_batch_size}")
lr = args.train_args.lr					#初始学习率
size = args.train_args.image_size			#图像尺寸
epochs = args.train_args.epochs				#总的epochs
snap_num = args.train_args.snap_num			#快照个数
weight_decay = args.train_args.weight_decay		#优化器的正则参数
resize_scale = args.train_args.resize_scale		#随机裁切的resize scale
erasing_prob = args.train_args.erasing_prob		#随机擦除的概率
using_cutmix = True if args.train_args.cutmix == 'True' else False			#是否开启cutmix
using_label_smooth = True if args.train_args.label_smooth == 'True' else False		#是否开启labelsmooth
model_path = args.train_args.model_path			#前一阶段训练得到的模型路径
classes = args.train_args.num_class        #需要识别的图像种类
log_path = args.train_args.log_path # 日志存储路径
# summary_writer = tf.summary.create_file_writer(log_path)
use_gpu = torch.cuda.is_available()

if using_label_smooth == True:
	criterion = loss.CrossEntropyLabelSmooth(10, epsilon=0.1)
else:
	criterion = nn.CrossEntropyLoss()
	
#训练中记录三个指标：验证集的最佳acc和对应的loss，验证集上的最低loss
metrics = {'best_acc':10, 'best_acc_loss':100, 'best_loss':100}

train_transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.RandomChoice([transforms.RandomCrop(size, padding=1, pad_if_needed=True, padding_mode='edge'),
                                                transforms.RandomResizedCrop(size, scale=(resize_scale, 1.0), ratio=(0.8, 1.2))]),
                    transforms.RandomHorizontalFlip(),
                    auto_augment.AutoAugment(dataset='CIFAR'),	#auto_augment
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=erasing_prob),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
val_transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df['FileID'], train_df['SpeciesID'])):
    print('*******************fold {0}*****************'.format(fold_idx))
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    # print(f"train_data:{train_data}")
    # print(val_data['FileID'])
    
    train_loader = dataset.DataLoaderX(dataset.QRDataset(list(train_data['FileID']), list(train_data['SpeciesID']), train_transform), 
                                batch_size=train_batch_size, 
                                shuffle=True, 
                                pin_memory=True)
    val_loader = dataset.DataLoaderX(dataset.QRDataset(list(val_data['FileID']), list(val_data['SpeciesID']), val_transform), 
                                batch_size=train_batch_size, 
                                shuffle=False, 
                                pin_memory=True)

    model = mymodel.Net_b5(num_class=classes)
    # model = mymodel.Net_multi_model(num_class=classes)
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print('load {}'.format(model_path))
    if use_gpu:
        model_ft = model.cuda()
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': lr})
            param.require_grad = True
            print(name)
        else:
            parameters.append({'params': param, 'lr': lr})
            param.require_grad = True
    optimizer = newoptim.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    # optimizer = ranger.Ranger(parameters, lr=lr, weight_decay=weight_decay)
    # 学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs//snap_num) #动态调整cos周期的 温和的cos 学习调整函数
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//snap_num) #余弦退火学习率调整
    current_snapshot = 0
    snapshots_losses = np.zeros((snap_num, 2))

    for epoch in range(epochs):
        print('Epoch: ', epoch)
        current_snapshot, metrics = reset_metrics(metrics, epoch, epochs, current_snapshot, snap_num)

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, scheduler, mixup=using_cutmix)
        # train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, scheduler, mixup=False)
        # with summary_writer.as_default():
        #     tf.summary.scalar('train_loss_T', train_loss, step=epoch)
        #     tf.summary.scalar('train_acc_T', train_acc, step=epoch)
        val_acc, val_loss = validate(val_loader, model, criterion)
#         with summary_writer.as_default():
#              tf.summary.scalar('val_loss_T', val_loss, step=epoch)
#              tf.summary.scalar('val_acc_T', val_acc, step=epoch)
        save_checkpoint(metrics, val_loss, val_acc, model, current_snapshot, fold_idx)
        snapshots_losses[current_snapshot][0] = metrics['best_acc']
        snapshots_losses[current_snapshot][1] = metrics['best_acc_loss']
        print('train_loss: {:4f} train_acc: {:4f}'.format(train_loss, train_acc))
        print('val_loss:   {:4f} val_acc:   {:4f}'.format(val_loss, val_acc))
        print('best loss:  {:4f} best acc:  {:4f} lr: {}'.format(metrics['best_loss'], metrics['best_acc'], optimizer.param_groups[0]['lr']))
        # with summary_writer.as_default():
        #     tf.summary.scalar('best_acc', metrics['best_acc'], step=epoch)
    best_acc_snap = snapshots_losses[np.where(snapshots_losses[:,0] == np.max(snapshots_losses))]
    best = best_acc_snap[np.where(best_acc_snap[:, 1] == np.min(best_acc_snap))]
    best_num = np.where(snapshots_losses[:, 1] == best[0][1])[0][0]
    model.load_state_dict(torch.load('./checkpoint/fold%d_snap%d.pth' % (fold_idx, best_num)))
    # model.load_state_dict(torch.load('./checkpoint_ranger/fold%d_snap%d.pth' % (fold_idx, best_num)))
    if using_cutmix == True:
        torch.save(model.state_dict(), './checkpoint/best_model_%d.pth' % size)
        # torch.save(model.state_dict(), './checkpoint_ranger/best_model_%d.pth' % size)
        print('save best_model_{}'.format(size))
    else:
        torch.save(model.state_dict(), './checkpoint/best_model_final.pth')
        # torch.save(model.state_dict(), './checkpoint_ranger/best_model_final.pth')
        print('save best_model_final')
    break
