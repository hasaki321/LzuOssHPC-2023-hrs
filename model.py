import torch.nn as nn
import torch
import torch.nn.functional as F



class ResBlock(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, down_sample=False):

        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch * self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch * self.expansion)
        )
        self.down_sample = down_sample
        if down_sample:
            self.down_sample_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch * self.expansion),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.down_sample:
            x = self.down_sample_layer(x)
            out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):    
    def __init__(self, out_features):
        super().__init__()

        self.in_ch = 64

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, self.in_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1 = self.get_layer(64, 3, False)
        self.conv2 = self.get_layer(128, 3, True)
        self.conv3 = self.get_layer(256, 5, True)
        self.conv4 = self.get_layer(512, 2, True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, out_features)

    def get_layer(self, channel, num_block, down_sample=False):
        layers = []
        block = ResBlock
        if down_sample:
            layers.append(block(self.in_ch, channel, 2, down_sample))
        else:
            layers.append(block(self.in_ch, channel))
        self.in_ch = channel * block.expansion

        for i in range(num_block):
            layers.append(block(self.in_ch, channel))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.pre_conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layer):
        super().__init__()
        net = [self.get_layer(out_ch, out_ch) for _ in range(1,num_layer)]
        net.insert(0,self.get_layer(in_ch, out_ch))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net = nn.Sequential(*net)

    def get_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return self.pool(x)


class VGG(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = self.get_block(3, 64, 2)
        self.conv2 = self.get_block(64, 128, 2)
        self.conv3 = self.get_block(128, 256, 3)
        self.conv4 = self.get_block(256, 512, 3)
        self.conv5 = self.get_block(512, 512, 3)
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_features),
        )

    def get_block(self, in_ch, out_ch, num_layer):
        return VGGBlock(in_ch, out_ch, num_layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = torch.flatten(out,1)

        out = self.fc(out)
        return out


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pre_conv = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.aux1 = InceptionAux(512, num_classes)  
        self.aux2 = InceptionAux(528, num_classes) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):  
        x = self.pre_conv(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training: 
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training:  
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training:
            return x, aux2, aux1
        return x


class Inception(nn.Module): 
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(  
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1) 
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        output = torch.cat(outputs, 1)
        return self.relu(output)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1) 

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x