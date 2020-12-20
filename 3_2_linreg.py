"""
线性回归的从零开始实现
https://tangshusen.me/Dive-into-DL-PyTorch/#/    网页版图书
https://github.com/ShusenTang/Dive-into-DL-PyTorch
"""

import torch
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
import random

# ###########################################################################
# 生成数据集
# ###########################################################################
num_inputs = 2  # 每个样本的特征数
num_examples = 1000  # 样本总数

true_w = [2, -3.4]  # 回归模型的真实权重
true_b = 4.2        # 回归模型的真实偏差

features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))  # 将numpy类转换成tensor类

labels = features[:, 0]*true_w[0] + features[:, 1]*true_w[1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))   # 生成样本的标签


# ###########################################################################
# 数据集可视化
# ###########################################################################
def use_svg_display():
    display.set_matplotlib_formats('svg')  # 用矢量图表示


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize  # 设置图的尺寸

'''
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)  # 数据集可视化， 观察标签和特征间的线性关系
plt.show()
'''


# ###########################################################################
# 读取数据
# ###########################################################################
def data_iter(batch_size, features, labels):    # 搞不清楚的时候可以到Python的IDLE里用具体的数字代入然后打印观察结果来理解
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机打乱indices里面的数的排列
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])  # 注意：最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
'''
for x, y in data_iter(batch_size, features, labels):
    print(x, y)
    # break
'''

# ###########################################################################
# 初始化模型参数
# ###########################################################################
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)  # 修改属性，可以对它们自动求梯度
b.requires_grad_(requires_grad=True)


# ###########################################################################
# 定义模型
# ###########################################################################
def linreg(x, w, b):
    x = torch.as_tensor(x, dtype=torch.float32)    # 少了会报错，警告 https://zhuanlan.zhihu.com/p/90590957 ,as_tensor()
    # x = x.clone().detach()
    return torch.mm(x, w) + b


# ###########################################################################
# 定义损失函数
# ###########################################################################
def squared_loss(y_hat, y):
    return (y_hat-y.view(y_hat.size())) ** 2 / 2  # 平方损失


# ###########################################################################
# 定义优化算法
# ###########################################################################
def sgd(params, lr, batch_size):   # 小批量随机梯度下降法
    for param in params:
        param.data -= lr * param.grad / batch_size  # 自动求出来的梯度是一个批量样本的梯度和


# ###########################################################################
# 训练模型
# ###########################################################################
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):  # x:特征， y:标签
        l = loss(net(x, w, b), y).sum()  # 小批量x和y的损失
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()  # 梯度清零
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))  # item(), 将一个标量Tensor转换成一个Python number

# ###########################################################################
# 学习到的参数和真实参数比较
# ###########################################################################
print(true_w, '\n', w)
print(true_b, '\n', b)
