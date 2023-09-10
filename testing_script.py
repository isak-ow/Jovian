import os
import tarfile
from torch.utils.data import DataLoader
import resnet9 as f
import resnet18 as g
import shakes_resnet18 as shakes
from torchvision.datasets import ImageFolder
import utils as u
import torch
import torch.nn as nn
import torchvision.transforms as tt

from torchvision.datasets.utils import download_url
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
if not os.path.exists('./data/cifar10'):
    download_url(dataset_url, '.')
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')
    
# Look into the data directory
data_dir = './data/cifar10'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
num_classes = len(classes)
color_channels = 3

print(classes)

valid_data = ImageFolder(data_dir+'/test', tt.ToTensor())

test_dl = DataLoader(valid_data,1000, num_workers=1)

device = u.get_default_device()

test_loader = u.DeviceDataLoader(test_dl,device)
model = torch.load('model.pth')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

model.eval()
for batch in enumerate(test_loader):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    acc = accuracy(output,labels)
    print('Accuracy of model:', acc)
    break


del model, images, labels
torch.cuda.empty_cache()