import torch
import wandb
import torch.nn as nn
import torchvision.transforms as tt

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True),
                         tt.RandomHorizontalFlip(), 
                         tt.RandomCrop(32, padding=4, padding_mode='reflect')
]) 
                         #tt.RandomRotation(30),
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
  
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train(model,train_loader,optimizer,criterion,scheduler,device):
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

def test(epoch,model,test_loader,criterion,device):
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