"""
3.7 SOFTMAX回归的简洁实现
"""

import torch
import torch.nn as nn
from torch.nn import init
import torchvision

import sys
from collections import OrderedDict

import d2lzh_pytorch as d2l


# #################################################################
# 获取和读取数据
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

# #################################################################
# 定义和初始化模型
# #################################################################
num_inputs = 784  # 28x28 为图像的高和宽
num_outputs = 10  # 手写数字为0-9，共10个类别


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


# net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# #################################################################
# SOFTMAX和交叉熵损失函数
# #################################################################
loss = nn.CrossEntropyLoss()

# #################################################################
# 定义优化算法
# #################################################################
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# #################################################################
# 训练模型
# #################################################################
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              batch_size, None, None, optimizer)


