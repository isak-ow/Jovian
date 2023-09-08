import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.RandomRotation(30),
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

class cifar_10_model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = ComplexResidualBlock(in_channels, 64)
        self.conv2 = ComplexResidualBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ComplexResidualBlock(128, 128), ComplexResidualBlock(128, 128))
        
        self.conv3 = ComplexResidualBlock(128, 256, pool=True)
        self.conv4 = ComplexResidualBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ComplexResidualBlock(512, 512), ComplexResidualBlock(512, 512))
        
        # self.conv5 = conv_block(512, 512, pool=True)
        # self.conv6 = conv_block(512, 512, pool=True)
        # self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        #input [batch,3,32,32]
        out = self.conv1(xb)
        #out [batch,64,32,32]
        out = self.conv2(out)
        #out [batch,128,16,16]
        out = self.res1(out) + out
        #out [batch,128,16,16]
        out = self.conv3(out)
        #out [batch,256,16,16]
        out = self.conv4(out)
        #out [batch,512,4,4]
        out = self.res2(out) + out
        #out [batch,512,4,4]
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.res3(out) + out
        print(out.shape)
        out = self.classifier(out)
        print(out.shape)
        return out

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# class cifar_10_model(nn.Module):
#     def __init__(self, in_channels, num_classes, dropout_rate=0.2):
#         super().__init__()
        
#         self.block_config = [
#             {'in': in_channels, 'out': 64, 'pool': False},
#             {'in': 64, 'out': 128, 'pool': True},
#             {'in': 128, 'out': 256, 'pool': True},
#             {'in': 256, 'out': 512, 'pool': True},
#         ]
        
#         self.blocks = nn.ModuleList()
#         for config in self.block_config:
#             self.blocks.append(conv_block(config['in'], config['out'], config.get('pool', False)))
        
#         self.res_blocks = nn.ModuleList([
#             nn.Sequential(conv_block(128, 128), conv_block(128, 128)),
#             nn.Sequential(conv_block(512, 512), conv_block(512, 512))
#         ])

#         self.complex_res_blocks = nn.ModuleList([
#             nn.Sequential(ComplexResidualBlock(128, 128), ComplexResidualBlock(128, 128)),
#             nn.Sequential(ComplexResidualBlock(512, 512), ComplexResidualBlock(512, 512)),
#             nn.Sequential(ComplexResidualBlock(512, 512), ComplexResidualBlock(512, 512))
#         ])

#         self.simple_res_blocks = nn.ModuleList([
#             nn.Sequential(SimpleResidualBlock(128, 128), SimpleResidualBlock(128, 128)),
#             nn.Sequential(SimpleResidualBlock(512, 512), SimpleResidualBlock(512, 512))
#         ])
        
#         self.classifier = nn.Sequential(
#             nn.MaxPool2d(4),
#             nn.Flatten(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, xb):
#         out = xb
#         for i, block in enumerate(self.blocks):
#             out = block(out)
#             if i == 1:  # After the second block
#                 out = self.complex_res_blocks[0](out) + out
#             if i == 3:  # After the fourth block
#                 out = self.complex_res_blocks[1](out) + out
#         out = self.classifier(out)
#         return out
    
class SimpleResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
    
class ComplexResidualBlock(nn.Module):
    # Example of a more complex residual block
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

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