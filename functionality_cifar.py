import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         #tt.RandomRotation(30),
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
def ResNet18(img_channel=3, num_classes=10):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes)


# class cifar_10_model(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
        
#         self.conv1 = conv_block(in_channels, 64)
#         self.conv2 = conv_block(64, 128, pool=True)
#         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
#         self.conv3 = conv_block(128, 256, pool=True)
#         self.conv4 = conv_block(256, 512, pool=True)
#         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
#         # self.conv5 = conv_block(512, 512, pool=True)
#         # self.conv6 = conv_block(512, 512, pool=True)
#         # self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

#         self.classifier = nn.Sequential(nn.MaxPool2d(4), 
#                                         nn.Flatten(), 
#                                         nn.Dropout(0.2),
#                                         nn.Linear(512, num_classes))
        
#     def forward(self, xb):
#         #input [batch,3,32,32]
#         out = self.conv1(xb)
#         #out [batch,64,32,32]
#         out = self.conv2(out)
#         #out [batch,128,16,16]
#         out = self.res1(out) + out
#         #out [batch,128,16,16]
#         out = self.conv3(out)
#         #out [batch,256,16,16]
#         out = self.conv4(out)
#         #out [batch,512,4,4]
#         out = self.res2(out) + out
#         #out [batch,512,4,4]
#         # out = self.conv5(out)
#         # out = self.conv6(out)
#         # out = self.res3(out) + out
#         print(out.shape)
#         out = self.classifier(out)
#         print(out.shape)
#         return out

# def conv_block(in_channels, out_channels, pool=False):
#     layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
#               nn.BatchNorm2d(out_channels), 
#               nn.ReLU(inplace=True)]
#     if pool: layers.append(nn.MaxPool2d(2))
#     return nn.Sequential(*layers)

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
    def __init__(self, in_channels, out_channels, pool=False):
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