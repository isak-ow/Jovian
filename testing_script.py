## running inference on the model trained on the cluster
import torch
import utils as u
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

device = u.get_default_device()
model = torch.load('model.pth',map_location=device)
data_dir = './data/cifar10'
test_data = ImageFolder(data_dir+'/test', transforms.ToTensor())
print(test_data.shape)

#loading test_set onto GPU
test_dl = DataLoader(test_data, 1000, num_workers=1, pin_memory=True)
test_loader = u.DeviceDataLoader(test_dl,device)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

model.eval()
for images, labels in enumerate(test_loader):
    print(images.shape)
    images = images.to(device)
    output = model(images)
    acc = accuracy(output,labels)
    print('Accuracy of model:', acc)
    break


del model, images, labels
torch.cuda.empty_cache()