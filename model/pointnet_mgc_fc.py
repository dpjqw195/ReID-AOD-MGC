import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from Anchor import GlobalModule, BasePointNet
import torch.nn.functional as F
from iter_pool import Pool

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def ball_query(xyz, new_xyz, radius, K):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

def sample_and_group(xyz, points, new_xyz, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    # new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)
    # grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()

        self.L = nn.Linear(indim, outdim, bias = False)   #25600*100
        self.L.requires_grad = False

        self.class_wise_learnable_norm = True

        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <=200:
            self.scale_factor = 2;
        else:
            self.scale_factor = 10;

    def forward(self, x):

        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor* (cos_dist)

        return scores

class PointNet_SA_Module(nn.Module):
    def __init__(self,radius, K, in_channels, mlp, group_all, bn=True, use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()

        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
        self.pool = Pool([[4, 128], [2, 64], [1, 32]])
    def forward(self, xyz, points, new_xyz):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              new_xyz=new_xyz,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.pool(new_points)
        return new_xyz, new_points



class pointnet_mgc(nn.Module):
    def __init__(self, in_channels, frame=200):
        super(pointnet_mgc, self).__init__()
        self.base = BasePointNet()
        self.glob1 = GlobalModule(30)
        self.pt_sa1 = PointNet_SA_Module(radius=20, K=8, in_channels=in_channels, mlp=[64, 64, 128, 256], group_all=False)
        self.feature = distLinear(frame *672+int(frame/2)*672+int(frame/4)*672, 256)

    def forward(self, xyz, points):
        batchsize, t, n, c = xyz.shape[0], xyz.shape[1], xyz.shape[2], xyz.shape[3]
        xyz = xyz.reshape(batchsize*t, n, c)
        if points is not None:
            points = points.reshape(batchsize*t, n, -1)

        new_xyz = self.base(xyz)
        _, new_xyz, _ = self.glob1(new_xyz, batchsize, t)

        new_xyz, new_points = self.pt_sa1(xyz, points, new_xyz)

        new_points = torch.max(new_points, dim=1)[0]
        net = new_points.reshape(batchsize, t, -1)
        net1 = F.max_pool2d(net, [2, 1])
        net2 = F.max_pool2d(net, [4, 1])
        net1 = net1.reshape(batchsize, -1)
        net2 = net2.reshape(batchsize, -1)
        net = net.reshape(batchsize, -1)

        net_res = torch.cat([net,net1,net2],1)


        net_res = self.feature(net_res)

        return net_res





if __name__ == '__main__':
    xyz = torch.randn(8,200,50, 3)
    # pt_sa = PointNet_SA_Module(radius=20, K=8, in_channels=3, mlp=[64, 64, 128, 256], group_all=False)
    net = pointnet_mgc(3)
    print(net(xyz, None).shape)


    # net = ssg_model(xyz, points)
    # print(net.shape)
    # print(label.shape)
    # loss = cls_loss()
    # loss = loss(net, label)
    # print(loss)