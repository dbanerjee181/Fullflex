import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

momentum = 0.999

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False, momentum=0.999, sync_bn=False):
        super(BasicBlock, self).__init__()
        if sync_bn:
            self.bn1 = nn.SyncBatchNorm(in_planes, momentum=momentum, eps=0.001)  # Use SyncBatchNorm if needed
        else:
            self.bn1 = nn.BatchNorm2d(in_planes, momentum=momentum, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        if sync_bn:
            self.bn2 = nn.SyncBatchNorm(out_planes, momentum=momentum, eps=0.001)  # Use SyncBatchNorm if needed
        else:
            self.bn2 = nn.BatchNorm2d(out_planes, momentum=momentum, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) if not self.equalInOut else None
        self.activate_before_residual = activate_before_residual

    def forward(self,x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)  # Use F.dropout in PyTorch
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)  # Use torch.add

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False, momentum=0.999, sync_bn=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, momentum, sync_bn)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, momentum, sync_bn):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes,
                                stride if i == 0 else 1, drop_rate, activate_before_residual, momentum=momentum, sync_bn=sync_bn))
        return nn.Sequential(*layers)  # Use nn.Sequential

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False, bn_momentum=0.999, sync_bn=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6  # Use // for integer division
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True, momentum=bn_momentum, sync_bn=sync_bn)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate, momentum=bn_momentum, sync_bn=sync_bn)        
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate, momentum=bn_momentum, sync_bn=sync_bn)
                if sync_bn:
            self.bn1 = nn.SyncBatchNorm(channels[3], momentum=momentum, eps=0.001)
        else:
            self.bn1 = nn.BatchNorm2d(channels[3], momentum=bn_momentum, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Use kaiming_normal_
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)  # Use constant_
                nn.init.constant_(m.bias, 0)  # Use constant_
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)  # Use constant_ 
                
    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)  # Use F.adaptive_avg_pool2d in PyTorch
        out = out.view(-1, self.channels)    # Use view instead of reshape
        output = self.fc(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                return output
            