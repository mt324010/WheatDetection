import torch
import torch.nn as nn

def conv1x1(in_channel, out_channel, stride):
    return nn.Conv2d(in_channel,out_channel,1,stride=stride)
def conv3x3(in_channel, out_channel, stride=1, padding=1):
    return nn.Conv2d(in_channel,out_channel,3,stride,padding)

class YOLO(nn.Module):

    def __init__(self, block, config, n_classes, n_anchors=3):
        """

        :param block: 'BottleNeck'
        :param config: like [1,3,4,6,3]
        :param n_classes: output shape will be like [batch, 3*5+n_classes, n*13, n*13]
        :param n_anchors: take the last n_anchor layers as output
        """
        super(YOLO, self).__init__()
        assert len(config) == 5
        self.base = 64
        self.input_shape = 64
        self.conv_anchor1 = conv1x1(self.base*4*4, 3*5+n_classes,1)
        self.conv_anchor2 = conv1x1(self.base*8*4, 3*5+n_classes,1)
        self.conv_anchor3 = conv1x1(self.base*16*4, 3*5+n_classes,1)
        self.conv1 = conv1x1(3, self.base,stride=1)
        self.layer1 = self._make_layer(block, config[0])
        self.layer2 = self._make_layer(block, config[1])
        self.layer3 = self._make_layer(block, config[2])
        self.layer4 = self._make_layer(block, config[3])
        self.layer5 = self._make_layer(block, config[4])


    def _make_layer(self, block, n):
        blocks = []
        for i in range(n):
            if i == 0:
                blocks.append(eval(block)(self.input_shape, 4*self.base, stride=2))
            else:
                blocks.append(eval(block)(4*self.base, 4*self.base, stride=1))
        self.base = self.base*2
        self.input_shape = 2*self.base
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out.append(self.conv_anchor1(x))
        x = self.layer4(x)
        out.append(self.conv_anchor2(x))
        x = self.layer5(x)
        out.append(self.conv_anchor3(x))
        return out

class BottleNeck(nn.Module):

    def __init__(self, in_channel, out_channel, stride):

        super(BottleNeck, self).__init__()
        if in_channel != out_channel:
            self.conv0 = conv1x1(in_channel, out_channel, stride)
        else:
            self.conv0 = None
        self.conv1=conv1x1(in_channel, out_channel//4,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel//4)
        self.conv2=conv3x3(out_channel//4, out_channel//4)
        self.bn2 = nn.BatchNorm2d(out_channel//4)
        self.conv3=conv1x1(out_channel//4, out_channel,stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    def forward(self, x):
        identity = x
        if self.conv0:
            identity = self.conv0(identity)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x += identity
        return self.activation(x)