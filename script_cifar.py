import os
import tarfile
from torch.utils.data import DataLoader
import resnet9 as f
import resnet18 as g
import shakes_resnet18 as shakes
from torchvision.datasets import ImageFolder
import wandb
import utils as u
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

wandb.init(project="cluster_CIFAR10_0809", name="more_blocks")
wandb.config.update({"architecture": "cifar10model", "dataset": "CIFAR-10", "epochs": 35, 
                     "batch_size": 128, "weight_decay": 5e-4, "max_lr": 0.1, "grad_clip": 1.5})

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

train_data = ImageFolder(data_dir+'/train', u.train_tfms)
valid_data = ImageFolder(data_dir+'/test', u.valid_tfms)

train_dl = DataLoader(train_data, wandb.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dl = DataLoader(valid_data, wandb.config.batch_size*2, num_workers=2, pin_memory=True)

device = u.get_default_device()

train_loader = u.DeviceDataLoader(train_dl,device)
test_loader = u.DeviceDataLoader(test_dl,device)

model = u.to_device(shakes.ResNet18(), device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=wandb.config.max_lr, 
                            weight_decay=wandb.config.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, wandb.config.max_lr, 
                                                epochs=wandb.config.epochs, 
                                                steps_per_epoch=len(train_loader))
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
best_acc = 0

# Create checkpoint directory if it doesn't exist
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

# Training loop
for epoch in range(wandb.config.epochs):
    u.train(model,train_loader,optimizer,criterion,scheduler,device)
    u.test(epoch,model,test_loader,criterion,device,best_acc)
    wandb.log({"epoch": epoch, "learning_rate": u.get_lr(optimizer)})  

wandb.finish()
torch.save(model, 'model.pth')

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
print(accuracy)
torch.cuda.empty_cache()