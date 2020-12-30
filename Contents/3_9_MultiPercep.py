"""
3.9 多层感知机的从零开始实现
"""

import torchvision
import torch
import sys

import numpy as np

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
# 定义模型参数
# #################################################################
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)

W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# #################################################################
# 定义激活函数
# #################################################################
def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))


# #################################################################
# 定义模型
# #################################################################
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# #################################################################
# 定义损失函数
# #################################################################
loss = torch.nn.CrossEntropyLoss()

# #################################################################
# 训练模型
# #################################################################
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              batch_size, params, lr)
