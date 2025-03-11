import numpy as np
import torch
import torch.nn as nn


class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3,   out_channels=8,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)
        x = torch.cat((in_mat[:,:,:3], x), -1)

        return x



class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class GlobalRNN(nn.Module):
    def __init__(self, M):
        super(GlobalRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(256,128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, M*3)

    def forward(self, x):
        g_vec, (hn, cn)=self.rnn(x)
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn

class GlobalModule(nn.Module):
    def __init__(self, M):
        super(GlobalModule, self).__init__()
        self.gpointnet=GlobalPointNet()
        self.grnn=GlobalRNN(M)
        self.M = M

    def forward(self, x, batch_size, length_size):
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x)
        g_loc = g_loc.reshape(batch_size*length_size, self.M, 3)
        return g_vec, g_loc, attn_weights


if __name__ == '__main__':

    # x = torch.randn(4,50,3)
    # net = BasePointNet()
    # print(net(x).shape)

    x = torch.randn(4,15,27)
    net = GlobalModule(15)
    g,l,_=net(x,2,2)
    print(g.shape)
    print(l.shape)