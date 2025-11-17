import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """Basic convolution block with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Block35(nn.Module):
    """Inception-ResNet-A block (35x35 grid)"""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(256, 32, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, 1),
            BasicConv2d(32, 32, 3, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, 1),
            BasicConv2d(32, 32, 3, padding=1),
            BasicConv2d(32, 32, 3, padding=1)
        )
        
        self.conv2d = nn.Conv2d(96, 256, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    """Inception-ResNet-B block (17x17 grid)"""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(896, 128, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, 1),
            BasicConv2d(128, 128, (1, 7), padding=(0, 3)),
            BasicConv2d(128, 128, (7, 1), padding=(3, 0))
        )
        
        self.conv2d = nn.Conv2d(256, 896, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    """Inception-ResNet-C block (8x8 grid)"""
    def __init__(self, scale=1.0, no_relu=False):
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        
        self.branch0 = BasicConv2d(1792, 192, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, 1),
            BasicConv2d(192, 192, (1, 3), padding=(0, 1)),
            BasicConv2d(192, 192, (3, 1), padding=(1, 0))
        )
        
        self.conv2d = nn.Conv2d(384, 1792, 1, bias=True)
        if not self.no_relu:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.no_relu:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """Reduction-A block (35x35 -> 17x17)"""
    def __init__(self):
        super().__init__()
        
        self.branch0 = BasicConv2d(256, 384, 3, stride=2)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, 1),
            BasicConv2d(192, 192, 3, padding=1),
            BasicConv2d(192, 256, 3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), 1)


class Mixed_7a(nn.Module):
    """Reduction-B block (17x17 -> 8x8)"""
    def __init__(self):
        super().__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 384, 3, stride=2)
        )
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 256, 3, stride=2)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, stride=2)
        )
        
        self.branch3 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class InceptionResnetV1(nn.Module):
    """Inception ResNet V1 for face recognition (160x160 input)"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Stem
        self.conv2d_1a = BasicConv2d(3, 32, 3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, 3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, 1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, 3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, 3, stride=2)
        
        # Inception-ResNet-A blocks (5x)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        
        # Reduction-A
        self.mixed_6a = Mixed_6a()
        
        # Inception-ResNet-B blocks (10x)
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        
        # Reduction-B
        self.mixed_7a = Mixed_7a()
        
        # Inception-ResNet-C blocks (5x)
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        
        # Single Block8 (the one in state_dict)
        self.block8 = Block8(scale=1.0, no_relu=True)
        
        # Classification block
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        self.last_linear = nn.Linear(1792, embedding_dim, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.1)
    
    def forward(self, x):
        # Stem
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        
        # Inception blocks
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        
        # Classification
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        x = self.last_bn(x)
        
        return x
