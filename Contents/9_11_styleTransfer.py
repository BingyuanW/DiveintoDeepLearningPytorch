"""
9.11 样式迁移
https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/docs/chapter09_computer-vision/9.11_neural-style.md

"""

import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 内容图像
content_img = Image.open('Sea.jpg')

# 样式图像
style_img = Image.open('StarryNight.jpg')

rgb_mean = np.array([0.485, 0.456, 0.406])  # 预训练模型所用图片数据的均值和标准差
rgb_std = np.array([0.229, 0.224, 0.225])


# 预处理图像
def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


# 后处理图像
def postprocess(img_tensor):  # 将输出图像中的像素值还原回标准化之前的值
    inv_normalize = torchvision.transforms.Normalize(
        mean= -rgb_mean / rgb_std,
        std= 1/rgb_std)
    to_PIL_image = torchvision.transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


# #################################################################
# 抽取特征
# #################################################################
# 使用基于ImageNet数据集预训练的VGG-19模型
pretrained_net = torchvision.models.vgg19(pretrained=True, progress=True)  # 第一次要下载vgg19模型参数
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 只保留VGG从输入层到最靠近输出层的内容层或样式层之间的所有层
net_list = []
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.features[i])
net = torch.nn.Sequential(*net_list)


# 逐层计算，并保留内容层和样式层的输出
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


# 对内容图像抽取内容特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


# 对样式图像抽取样式特征
def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


# #################################################################
# 定义损失函数
# #################################################################
# 内容损失，采用平方误差函数
def content_loss(Y_hat, Y):
    return F.mse_loss(Y_hat, Y)


# 格拉姆矩阵， 表达样式层输出的样式
def gram(X):
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]
    X = X.view(num_channels, n)
    return torch.matmul(X, X.t()) / (num_channels * n)


# 样式损失
def style_loss(Y_hat, gram_Y):
    return F.mse_loss(gram(Y_hat), gram_Y)


#  总变差损失，降噪
def tv_loss(Y_hat):
    return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))


# 损失函数，为内容损失、样式损失和总变差损失的加权和
content_weight, style_weight, tv_weight = 1, 0.7*1e6, 10  # 原来的参数  1，1e3，10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


# #################################################################
# 创建和初始化合成图像
# #################################################################
# 定义一个简单的模型
class GeneratedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super(GeneratedImage, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))

    # 返回模型参数
    def forward(self):
        return self.weight


# 合成图像的模型实例
def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


# #################################################################
# 训练模型
# #################################################################
def train(X, contents_Y, styles_Y, device, lr, max_epochs, lr_decay_epoch):
    print("training on ", device)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)
    for i in range(max_epochs):
        start = time.time()

        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time() - start))
    return X.detach()


image_shape = (520, 780)  # 生成的图片太大了显存会不够用，本人用的是 NVIDIA GeForce GTX 960
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
style_X, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 35000, 10000)

postprocess(output).save('out.png')  # 保存图片到本地

# d2l.plt.imshow(postprocess(output))
# d2l.plt.axis('off')
# d2l.plt.show()
# d2l.plt.imsave('new.png', postprocess(output))
