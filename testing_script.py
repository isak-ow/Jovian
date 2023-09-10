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

valid_data = ImageFolder(data_dir+'/test', transform=tt.ToTensor())

test_dl = DataLoader(valid_data,1000, num_workers=1)

device = u.get_default_device()

test_loader = u.DeviceDataLoader(test_dl,device)
model = torch.load('model.pth')

model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    _, prediction = torch.max(output.data,1)
    total += labels.size(0)
    correct += (prediction == labels).sum().item()

accuracy = correct/total*100


del model, images, labels
torch.cuda.empty_cache()