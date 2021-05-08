import torch.nn as nn
from MyEfficientNet import cbam_EfficientNet
from MyEfficientNet import EfficientNet
# from utils import args
import torch

# b5 with cbam
class Net_b5(nn.Module):
    def __init__(self, num_class):
        super(Net_b5, self).__init__()
        # 导入融合cbam的Efficientnet模型
        self.model = cbam_EfficientNet.from_pretrained('efficientnet-b5', num_classes=1000)
        # # 导入原始的Efficientnet模型
        # self.model = EfficientNet.from_pretrained('efficientnet-b5', num_class=1000)
        self.num_class = num_class
        num_ftrs = self.model._fc.in_features
        self.fc = nn.Linear(num_ftrs,num_class)

    def forward(self, img):
        out = self.model(img)
        out = self.fc(out)
        return out

# naive b5
class Original_b5(nn.Module):
    def __init__(self, num_class):
        super(Original_b5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1000)
        self.num_class = num_class
        num_ftrs = self.model._fc.in_features
        self.fc = nn.Linear(num_ftrs,num_class)

    def forward(self, img):
        out = self.model(img)
        out = self.fc(out)
        return out

# 多尺度融合模型b3 and b6
class Net_multi_model(nn.Module):
    def __init__(self, num_class):
        super(Net_multi_model, self).__init__()
        
        self.model1 = cbam_EfficientNet.from_pretrained('efficientnet-b6', num_classes=1000)
        self.model2 = cbam_EfficientNet.from_pretrained('efficientnet-b3', num_classes=1000)
        self.num_class = num_class
        num_ftrs = self.model1._fc.in_features + self.model2._fc.in_features
        # self.dropout = nn.Dropout(0.5)
        self.attention = nn.Sequential(nn.Linear(num_ftrs, num_ftrs//16),
                                nn.ReLU(),
                                nn.Linear(num_ftrs//16, num_ftrs),
                                nn.Sigmoid())
        self.fc = nn.Linear(num_ftrs, self.num_class)


    def forward(self, img):
        out1 = self.model1(img)
        out2 = self.model2(img)
        out = torch.cat((out1, out2), 1)
        atten = self.attention(out)
        # out = self.dropout(out)
        input = torch.mul(atten, out)
        out = self.fc(out)
        return out

