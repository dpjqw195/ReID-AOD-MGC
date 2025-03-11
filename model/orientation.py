import torch
import torch.nn as nn
import torch.nn.functional as F
from spp import SPPLayer
from iter_pool import Pool
def index_points(points, idx):
    """
    Description:
        this function select the specific points from the whole points according to the idx.
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def pointsift_select(radius, xyz):
    """
    code by python matrix logic
    :param radius:
    :param xyz:
    :return: idx
    """
    dev = xyz.device
    B, N, _ = xyz.shape
    judge_dist = radius ** 2
    idx = torch.arange(N).repeat(8, 1).permute(1, 0).contiguous().repeat(B, 1, 1).to(dev)
    for n in range(N):
        distance = torch.ones(B, N, 8).to(dev) * 1e10
        distance[:, n, :] = judge_dist
        centroid = xyz[:, n, :].view(B, 1, 3).to(dev)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # shape: (B, N)
        subspace_idx = torch.sum((xyz - centroid + 1).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev),
                                 -1)
        for i in range(8):
            mask = (subspace_idx == i) & (dist > 1e-10) & (dist < judge_dist)  # shape: (B, N)
            distance[..., i][mask] = dist[mask]
            idx[:, n, i] = torch.min(distance[..., i], dim=-1)[1]
    return idx


def pointsift_group(radius, xyz, points, use_xyz=True):
    B, N, C = xyz.shape
    assert C == 3
    idx = pointsift_select(radius, xyz)  # B, N, 8

    grouped_xyz = index_points(xyz, idx)  # B, N, 8, 3
    grouped_xyz -= xyz.view(B, N, 1, 3)
    if points is not None:
        grouped_points = index_points(points, idx)
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        grouped_points = grouped_xyz
    return grouped_xyz, grouped_points, idx


def pointsift_group_with_idx(idx, xyz, points, use_xyz=True):
    B, N, C = xyz.shape
    grouped_xyz = index_points(xyz, idx)  # B, N, 8, 3
    grouped_xyz -= xyz.view(B, N, 1, 3)
    if points is not None:
        grouped_points = index_points(points, idx)
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        grouped_points = grouped_xyz
    return grouped_xyz, grouped_points


class PointSIFT_module(nn.Module):
    def __init__(self, radius, output_channel):
        super(PointSIFT_module, self).__init__()
        self.radius = radius
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channel, [1, 1], [1, 1]),
            nn.Conv2d(output_channel, output_channel, [1, 1], [1, 1]),
            nn.Conv2d(output_channel, output_channel, [1, 1], [1, 1])
        )
        # self.spp = SPPLayer(4)
        self.spp = Pool([[4,4],[2,2],[1,1]])
        # self.spp = Pool([[4,8],[2,4],[1,1]])


    def forward(self, x):
        batchsize, t, n, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(batchsize*t, n, c)
        grouped_xyz, grouped_points, idx = pointsift_group(self.radius, x, None)
        grouped_points = grouped_points.permute(0,3,1,2)
        grouped_points = self.conv1(grouped_points)

        grouped_points = grouped_points.permute(0,2,3,1)
        grouped_points = self.spp(grouped_points)
        return grouped_points.reshape(batchsize, t, n, -1)



class PointSIFT_res_module(nn.Module):
    def __init__(self, radius, output_channel, extra_input_channel=3, merge='concat', same_dim=False):
        super(PointSIFT_res_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + extra_input_channel, output_channel, [1, 2], [1, 2]),
            nn.Conv2d(output_channel, output_channel, [1, 2], [1, 2]),
            nn.Conv2d(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3 + output_channel, output_channel, [1, 2], [1, 2]),
            nn.Conv2d(output_channel, output_channel, [1, 2], [1, 2]),
            nn.Conv2d(output_channel, output_channel, [1, 2], [1, 2])
        )
        if same_dim:
            self.convt = nn.Sequential(
                nn.Conv1d(extra_input_channel, output_channel, 1),
                nn.BatchNorm1d(output_channel),
                nn.ReLU()
            )

    def forward(self, xyz, points):
        batchsize, t, n, c = xyz.shape[0], xyz.shape[1], xyz.shape[2], xyz.shape[3]
        xyz = xyz.reshape(batchsize*t, n, c)
        points = points.reshape(batchsize*t,n,-1)
        _, grouped_points, idx = pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        ##print(new_points.shape)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        _, grouped_points = pointsift_group_with_idx(idx, xyz, new_points)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        ##print(grouped_points.shape)
        new_points = self.conv2(grouped_points)

        new_points = new_points.squeeze(-1)

        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
            # print(points.shape)
            if self.same_dim:
                points = self.convt(points)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = torch.cat([new_points, points], dim=1)

        new_points = F.relu(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()
        xyz = xyz.reshape((batchsize, t, n, c))
        new_points = new_points.reshape((batchsize, t, n, -1))

        return xyz, new_points

if __name__ == '__main__':
    x = torch.randn((2,100,50,3))
    net = PointSIFT_module(20,32)
    print(net(x).shape)
