"""
3.10 多层感知机的简洁实现
"""

import torch
import torch.nn as nn
from torch.nn import init
import torchvision

import sys

import d2lzh_pytorch as d2l

# #################################################################
# 定义模型
# #################################################################
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
    )

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


# #################################################################
# 读取数据并训练模型
# #################################################################
def load_data_fashion_mnist(batch_size, resize=None, root='~/DiveIntoDLPytorch/FashionMNIST/Datasets'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))

    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)

    if sys.platform.startswith('win'):
        num_workers = 0  # 不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5

d2l.train_ch3(net, train_iter, test_iter, loss,
              num_epochs, batch_size, None, None, optimizer)
