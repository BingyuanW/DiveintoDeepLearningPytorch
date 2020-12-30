"""
3.3线性回归的简洁实现

"""
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

import numpy as np
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
# 读取数据
# ###########################################################################
batch_size = 10
dataset = Data.TensorDataset(features, labels)   # 将特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机读取小批量数据

# for x, y in data_iter:
#    print(x, y)


# ###########################################################################
# 定义模型
# ###########################################################################
'''
class LinearNet(nn.Module):                      
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)   # 用这种方法定义模型，没有net[0]      
'''
net = nn.Sequential(nn.Linear(num_inputs, 1))
print(net[0])  # 打印网络结构

# ###########################################################################
# 初始化模型参数
# ###########################################################################
init.normal_(net[0].weight, mean=0, std=0.01)  # 初始化权重参数的每个元素，按均值0、标准差0.01 正态分布
init.constant_(net[0].bias, val=0)   # 初始化偏差为0

# ###########################################################################
# 定义损失函数
# ###########################################################################
loss = nn.MSELoss()

# ###########################################################################
# 定义优化算法
# ###########################################################################
optimizer = optim.SGD(net.parameters(), lr=0.03)
'''
optimizer = optim.SGD([{'params': net.subnet1.parameters()},                # 为不同的子网络设置不同的学习率
                       {'params': net.subnet2.parameters(), 'lr': 0.01}],
                      lr=0.03)

for param_group in optimizer.param_groups:  # 调整学习率
    param_group['lr'] *= 0.1
'''
print(optimizer)

# ###########################################################################
# 训练模型
# ###########################################################################
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for x, y in data_iter:
        x = x.clone().detach().float()  # https://zhuanlan.zhihu.com/p/90590957
        output = net(x)
        y = y.clone().detach().float()  #
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
