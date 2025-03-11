import torch
import torch.nn as nn
import math
class Pool(nn.Module):
    def __init__(self, k):
        super(Pool, self).__init__()
        self.k = k

    def forward(self, x):
        output = []
        num, c, h, w = x.size()
        for i in self.k:
            level_h = i[0]
            level_w = i[1]
            kernel_size = (math.ceil(h / level_h), math.ceil(w / level_w))
            stride = (math.ceil(h / level_h), math.ceil(w / level_w))
            pooling = (math.floor((kernel_size[0] * level_h - h + 1) / 2), math.floor((kernel_size[1] * level_w - w + 1) / 2))
            max_pool = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=pooling)
            x = max_pool(x)
            output.append(x.reshape(num, c, -1))
            h = level_h
            w = level_w

        for index, p in enumerate(output):
            if index == 0:
                x_flatten = p
            else:
                x_flatten = torch.cat((x_flatten, p),-1)

        return x_flatten

if __name__ == '__main__':
    x = torch.randn((2,50,8,256))
    # net = Pool([[4,4],[2,2],[1,1]])
    net = Pool([[4,32],[2,16],[1,1]])
    print(net(x).shape)