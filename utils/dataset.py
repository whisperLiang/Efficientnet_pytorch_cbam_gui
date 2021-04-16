import torch
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from prefetch_generator import BackgroundGenerator
from PIL import Image

#Dataset需要传入所有图像的路径以及对应的label
class QRDataset(Dataset):
    def __init__(self, filename, label, transform):
        self.filenames = filename
        self.labels = label
        self.transform = transform
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

#采用prefetch_generator对DataLoader进行包装，可以提高数据读取速度        
class DataLoaderX(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
