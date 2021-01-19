import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from lenet5 import Lenet5
from resNet import ResNet18
import time


def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cpu')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    optimier = optim.Adam(model.parameters(), lr=1e-3)

    print(type(cifar_train))
    print((enumerate(cifar_train)))
    startTime = time.time()
    keepList = []
    for epoch in range(40):
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # backPropagation
            optimier.zero_grad()
            loss.backward()
            optimier.step()
        #
        print(epoch, loss.item())
        total_correct = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct/total_num
            endTime = time.time()
            print(epoch, acc)
            keepList.append((endTime-startTime, acc))
            print('time:'+str(endTime - startTime))
            # startTime = time.time()
            if endTime - startTime > 20000:
                break
    file = open('resnet18.txt', 'w')
    for elem in keepList:
        cur = str(elem)
        file.write(cur+'\n')
    file.close()


if __name__ == '__main__':
    main()