from utils import tools, args, dataset, mymodel, loss
import torch
import pandas as pd
from torchvision import transforms
from torch import nn
from tqdm import tqdm
size = args.train_args.image_size
test_batch_size = args.train_args.batch_size
classes = args.train_args.num_class        #需要识别的图像种类
model_path = args.train_args.model_path			#前一阶段训练得到的模型路径

def validate(val_loader, model, criterion):
    batch_time = tools.AverageMeter('Time', ':6.3f')
    losses = tools.AverageMeter('Loss', ':.4e')
    top1 = tools.AverageMeter('Acc@1', ':2.2f')
    top5 = tools.AverageMeter('Acc@5', ':2.2f')
    progress = tools.ProgressMeter(len(val_loader), batch_time, losses)
    model.eval()

    with torch.no_grad():
        # end = time.time()
        for i, (input, target) in tqdm(enumerate(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = tools.accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # batch_time.update(time.time() - end)
            # end = time.time()

        return top1.avg.data.cpu().numpy(), losses.avg

val_transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])

# for colab

val_data = pd.read_csv('./af2020cv-2020-05-09-v5-dev/annotation.csv')
val_data["FileID"] = "test/" + val_data["SpeciesID"].astype(str)+ "/" + val_data["FileID"] + '.jpg'
val_loader = dataset.DataLoaderX(dataset.QRDataset(list(val_data['FileID']), list(val_data['SpeciesID']), val_transform),
                                batch_size=test_batch_size,
                                shuffle=True,
                                pin_memory=True)

# print(shuffle)
model = mymodel.Net_b5(num_class=classes)
if model_path:
        model.load_state_dict(torch.load(model_path))
        print('load {}'.format(model_path))
model = model.cuda()
criterion = loss.CrossEntropyLabelSmooth(10, epsilon=0.1)
# criterion = nn.CrossEntropyLoss()
val_acc, val_loss = validate(val_loader, model, criterion)
print('val_loss:   {:4f} val_acc:   {:4f}'.format(val_loss, val_acc))

