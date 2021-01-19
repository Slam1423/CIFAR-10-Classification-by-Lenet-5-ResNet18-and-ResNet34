import torch
from torch import nn


class Lenet5(nn.Module):
    # For cifar 10
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3,6,5,1,0),
            nn.AvgPool2d(2,2,0),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.AvgPool2d(2, 2, 0),
        )

        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print('tmp:', out.shape)
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        # param x: [b, 3, 32, 32]
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, -1)
        logits = self.fc_unit(x)
        # loss = self.criteon(logits, y)
        return logits


def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('tmp:', out.shape)


if __name__ == '__main__':
    main()