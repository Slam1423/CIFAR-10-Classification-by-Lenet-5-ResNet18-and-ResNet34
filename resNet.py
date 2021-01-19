import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.blk1 = ResBlk(64, 128, 2)
        self.blk2 = ResBlk(128, 256, 2)
        self.blk3 = ResBlk(256, 512, 2)
        self.blk4 = ResBlk(512, 512)

        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):
        """

        :param self:
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        out = self.blk1(x)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        # print('after conving:', out.shape)
        X = F.adaptive_avg_pool2d(out, [1, 1])
        # print('after pooling:', X.shape)
        X = X.view(X.size(0), -1)
        out = self.outlayer(X)
        return out


def main():
    blk = ResBlk(64, 128, stride=2)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet18:', out.shape)


if __name__ == '__main__':
    main()