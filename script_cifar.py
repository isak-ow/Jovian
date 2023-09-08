import os
import tarfile
from torch.utils.data import DataLoader
import functionality_cifar as f
from torchvision.datasets import ImageFolder
import wandb
import torch
import torch.nn as nn
import pre_act_model as pre

# epochs = 24
# batch_size = 400
# weight_decay = 1e-4
# max_lr = 0.01
# grad_clip = 0.1 

wandb.init(project="cluster_CIFAR10_0809", name="better_logging_1")
wandb.config.update({"architecture": "cifar10model", "dataset": "CIFAR-10", "epochs": 24, 
                     "batch_size": 400, "weight_decay": 1e-4, "max_lr": 0.01, "grad_clip": 0.1})

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="cluster_CIFAR10_0809",
#     name = "better_logging_1",
    
#     # track hyperparameters and run metadata
#     config={
#     "architecture": "cifar10model",
#     "dataset": "CIFAR-10",
#     "epochs": epochs,
#     }
# )

from torchvision.datasets.utils import download_url
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
if not os.path.exists('./data/cifar10'):
    download_url(dataset_url, '.')
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

# Extract from archive
# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')
    
# Look into the data directory
data_dir = './data/cifar10'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
num_classes = len(classes)
color_channels = 3

print(classes)

train_data = ImageFolder(data_dir+'/train', f.train_tfms)
valid_data = ImageFolder(data_dir+'/test', f.valid_tfms)

train_dl = DataLoader(train_data, wandb.config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_dl = DataLoader(valid_data, wandb.config.batch_size*2, num_workers=1, pin_memory=True)

device = f.get_default_device()

train_loader = f.DeviceDataLoader(train_dl,device)
test_loader = f.DeviceDataLoader(test_dl,device)

model = f.to_device(f.cifar_10_model(color_channels, num_classes), device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=wandb.config.max_lr, 
                            weight_decay=wandb.config.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, wandb.config.max_lr, 
                                                epochs=wandb.config.epochs, 
                                                steps_per_epoch=len(train_loader))
best_acc = 0

import os

# Create checkpoint directory if it doesn't exist
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        if wandb.config.grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), wandb.config.grad_clip)
        
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy})

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
        acc = 100. * correct / total
        
        if acc > best_acc:
            print('Saving best model...')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        
    avg_test_loss = test_loss / len(test_loader)
    wandb.log({"acc": acc, "test_loss": avg_test_loss, "best_acc": best_acc})

# Training loop
for epoch in range(wandb.config.epochs):
    train(epoch)
    test(epoch)
    wandb.log({"epoch": epoch, "learning_rate": f.get_lr(optimizer)})  

wandb.finish()
torch.save(model, 'model.pth')
torch.cuda.empty_cache()


# def train(epoch):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for i, batch in enumerate(train_loader):
#         images, labels = batch
#         images = images.to(device)
#         labels = labels.to(device)

#         if grad_clip: 
#             nn.utils.clip_grad_value_(model.parameters(), grad_clip)

#         outputs = model(images)
#         loss = criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
    
#     wandb.log({"train_loss": train_loss})

# def test(epoch):
#     global best_acc
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             images, labels = batch
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
                

#         acc = 100.*correct/total
#         wandb.log({"acc": acc})
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': model.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc
#         wandb.log({"best_acc": best_acc})
    

#     wandb.log({"test_loss": test_loss})

# torch.cuda.empty_cache() #removes any residual data from the gpus

# #training loop
# for epoch in range(epochs):
#     wandb.log({"epoch": epoch})
#     train(epoch)
#     test(epoch)
#     wandb.log({"learning_rate": f.get_lr(optimizer)})

# wandb.finish()
# torch.save(model,'model.pth')

