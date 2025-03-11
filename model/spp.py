import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):

        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, c, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, c, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, c, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, c, -1)), 2)
        return x_flatten

if __name__ == '__main__':

    x = torch.randn((4,50,8,32))
    num,c,h,w = 4,50,8,32
    # level = 2
    # kernel_size = (math.ceil(h / level), math.ceil(w / level))
    # stride = (math.ceil(h / level), math.ceil(w / level))
    # pooling = (math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
    spp = SPPLayer(3)
    print(spp(x).shape)
    # net = nn.MaxPool2d(kernel_size, stride, pooling)
    # print(net(x).shape)