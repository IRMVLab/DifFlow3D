import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
from pointnet2 import pointnet2_utils
from tqdm.auto import tqdm

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn,bias=True):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True, groups=1):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True, groups=1):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def L1_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.abs(torch.sum(src, -1).view(B, N, 1) - torch.sum(dst, -1).view(B, 1, M))
    return dist

def cosine_distance(src, dst):
    """
    Calculate cosine similarity distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    src = src / torch.sqrt(torch.sum(src ** 2, -1, keepdim=True) + 1e-8)
    dst = dst / torch.sqrt(torch.sum(dst ** 2, -1, keepdim=True) + 1e-8)
    dist = 1.0 - torch.bmm(src, dst.transpose(1, 2))

    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def knn_point_cosine(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = cosine_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def knn_point_l1(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = L1_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query_feat(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    # idx = knn_point(nsample, s_xyz, new_xyz)
    # grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    knn_idx = knn_point_cosine(nsample//2,  s_xyz, new_xyz) # B, N1, nsample
    knn_idx_p = knn_point(nsample//2, s_xyz, new_xyz) # B, N1, nsample
    grouped_xyz = torch.cat((index_points_group(s_xyz, knn_idx),index_points_group(s_xyz, knn_idx_p)),-2)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if s_points is not None:
        # grouped_points = index_points_group(s_points, idx)
        grouped_points =  torch.cat((index_points_group(s_points, knn_idx),
                                index_points_group(s_points, knn_idx_p)),-2)# B, N1, nsample, D2

        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

# add
def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

class SetAbstract(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, use_leaky = True):
        super(SetAbstract, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        if self.npoint == xyz.size(-1) or self.npoint is None:
            new_xyz = xyz
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  self.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points#, fps_idx

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True, use_act=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        #print(f'weightnet {weightnet}')
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.use_act = use_act
        if use_act:
            self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]
        #print(f'pointsshape {new_points.shape}')

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        #print(f'weights {weights.shape}')
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        #print(f'pointsshape {new_points.shape}')
        #print(f'pointsshape {new_points.shape}')
        #print(f'linearweight {self.linear.weight.shape}')
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        if self.use_act:
            new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points, fps_idx=None, new_xyz=None):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1)

        if new_xyz is None:
            if fps_idx is None:
                fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_gather(xyz, fps_idx)
        else:
            new_xyz = new_xyz.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        if fps_idx is not None:
            return new_xyz.permute(0, 2, 1), new_points, fps_idx 
        else:
            return new_xyz.permute(0, 2, 1), new_points

class CrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayer,self).__init__()
        # self.fe1_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel,in_channel], pooling=pooling, corr_func=corr_func)
        # self.fe2_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel, out_channel], pooling=pooling, corr_func=corr_func)
        # self.flow = nn.Conv1d(out_channel, 3, 1)

        self.nsample = nsample
        self.bn = bn
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()
        last_channel = in_channel  * 2 + 3
        for out_channel in mlp1:
            self.mlp1_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp1_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2 is not None:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            last_channel = mlp1[-1] * 2 + 3
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(mlp_convs):
            if self.bn:
                bn = mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        print(f'grouppoints1{feat1_new.shape}')
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2, cross_only = False):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False or cross_only:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final
    # def forward(self, pc1, pc2, feat1, feat2, bid=False):

    #     feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
    #     feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

    #     if self.mlp2 is False:
    #         return feat1_new, feat2_new

    #     feat1_final = self.cross(pc1, pc2, self.cross_t1(feat1_new), self.cross_t2(feat2_new), self.pos2, self.mlp2, self.bn2)
    #     if bid:
    #         feat2_final = self.cross(pc2, pc1, self.cross_t1(feat2_new), self.cross_t2(feat1_new), self.pos2, self.mlp2, self.bn2)
    #         return feat1_new, feat2_new, feat1_final, feat2_final

    #     return feat1_new, feat2_new, feat1_final

class CrossLayerLightFeat(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightFeat,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, knn1, knn2, pos, mlp, bn, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point(nsample//2,  knn2, knn1) # B, N1, nsample
        knn_idx_p = knn_point(nsample//2, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = torch.cat((index_points_group(xyz2, knn_idx),index_points_group(xyz2, knn_idx_p)),-2)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 =  torch.cat((index_points_group(points2, knn_idx).permute(0, 3, 2, 1),
                                      index_points_group(points2, knn_idx_p).permute(0, 3, 2, 1)),-2)# B, N1, nsample, D2
        # knn_idx = knn_point(nsample, xyz2, xyz1) # B, N1, nsample
        # neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2

        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, knn1, knn2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2, self.pos1, self.mlp1, self.bn1)
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), knn2, knn1, self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, knn1, knn2, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightFeatCosine(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightFeatCosine,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, knn1, knn2, pos, mlp, bn, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point_cosine(nsample//2,  knn2, knn1) # B, N1, nsample
        knn_idx_p = knn_point(nsample//2, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = torch.cat((index_points_group(xyz2, knn_idx),index_points_group(xyz2, knn_idx_p)),-2)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 =  torch.cat((index_points_group(points2, knn_idx).permute(0, 3, 2, 1),
                                      index_points_group(points2, knn_idx_p).permute(0, 3, 2, 1)),-2)# B, N1, nsample, D2
        # knn_idx = knn_point(nsample, xyz2, xyz1) # B, N1, nsample
        # neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2

        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, knn1, knn2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2, self.pos1, self.mlp1, self.bn1)
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), knn2, knn1, self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, knn1, knn2, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class BidirectionalLayerFeatCosine(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(BidirectionalLayerFeatCosine,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, knn1, knn2, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point_cosine(nsample//2,  knn2, knn1) # B, N1, nsample
        knn_idx_p = knn_point(nsample//2, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = torch.cat((index_points_group(xyz2, knn_idx),index_points_group(xyz2, knn_idx_p)),-2)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 =  torch.cat((index_points_group(points2, knn_idx).permute(0, 3, 2, 1),
                                      index_points_group(points2, knn_idx_p).permute(0, 3, 2, 1)),-2)# B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, knn1, knn2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2)

        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), knn2, knn1)

        return feat1_new, feat2_new

class BidirectionalLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(BidirectionalLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2))

        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1))

        return feat1_new, feat2_new

class BidirectionalLayerNeural(nn.Module):
    def __init__(self, nsample, in_channel, mlp, dist_out=128, dist_unit = None, bn = use_bn, use_leaky = True):
        super(BidirectionalLayerNeural, self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.dist = NeuralCosineDistance(mlp[0], dist_out, dist_unit)

    def cross(self, xyz1, xyz2, points1, points2, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        dist = self.dist(xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1), points1.permute(0, 2, 1), points2.permute(0, 2, 1))
        _, knn_idx = torch.topk(dist, nsample, dim = -1, largest=False, sorted=False)        
        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2))

        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1))

        return feat1_new, feat2_new

class BidirectionalLayerFuse(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(BidirectionalLayerFuse,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, knn1, knn2, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point_fuse(self.nsample, xyz2, xyz1, knn2, knn1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, knn1, knn2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2)

        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), knn2, knn1)

        return feat1_new, feat2_new

class FlowEmbeddingLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(FlowEmbeddingLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.conv1 = nn.Conv1d(in_channel, mlp[0], 1)
        self.conv2 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2, knn1, knn2, nsample=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape

        points1 = self.conv1(points1)
        points2 = self.conv2(points2)
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)

        if nsample is None:
            nsample=self.nsample

        knn_idx = knn_point_cosine(nsample//2,  knn2, knn1) # B, N1, nsample
        knn_idx_p = knn_point(nsample//2, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = torch.cat((index_points_group(xyz2, knn_idx),index_points_group(xyz2, knn_idx_p)),-2)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 =  torch.cat((index_points_group(points2, knn_idx).permute(0, 3, 2, 1),
                                      index_points_group(points2, knn_idx_p).permute(0, 3, 2, 1)),-2)# B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None, neighr=3):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(neighr, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, neighr, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 4, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        #flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        update = self.fc(new_points)
        #print(f'update {update.shape}')
        flow_local = update[:,:3,:].clamp(self.clamp[0], self.clamp[1]) 
        certainty = update[:,3:,:]
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        #print(f'flow: {flow.shape}')
        #print(f'points: {new_points.shape}')
        return new_points, flow, certainty

class GRUMappingNoGCN(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True, return_inter=False, relu=False, use_fuse=True):
        super(GRUMappingNoGCN,self).__init__()
        self.nsample = nsample
        self.use_fuse = use_fuse
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.bn = bn
        self.relu = relu

        last_channel = 3

        self.fuse_r = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(in_channel, mlp[0], 1, bias=False)

        self.fuse_r_2 = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.fuse_r_o_2 = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.fuse_z_2 = nn.Conv1d(in_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2, knn1, knn2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        knn1 = knn1.permute(0, 2, 1)
        knn2 = knn2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)

        points2_r = self.fuse_r_2(points2).permute(0, 2, 1)
        points2_r_o = self.fuse_r_o_2(points2).permute(0, 2, 1)
        points2_z = self.fuse_z_2(points2).permute(0, 2, 1)

        if self.nsample == N2 or self.nsample is None:
            grouped_points2_r = points2_r.unsqueeze(1).repeat(1,N1,1,1).permute(0, 3, 2, 1)
            grouped_points2_r_o = points2_r_o.unsqueeze(1).repeat(1,N1,1,1).permute(0, 3, 2, 1)
            grouped_points2_z = points2_z.unsqueeze(1).repeat(1,N1,1,1).permute(0, 3, 2, 1)
            direction_xyz = (xyz2.unsqueeze(1).repeat(1,N1,1,1) - xyz1.view(B, N1, 1, C)).permute(0, 3, 2, 1)
        else:
            if self.use_fuse:
                knn_idx = knn_point_cosine(self.nsample//2,  knn2, knn1) # B, N1, nsample
                knn_idx_p = knn_point(self.nsample//2, xyz2, xyz1) # B, N1, nsample
                idx = torch.cat((knn_idx, knn_idx_p), -1)
            else:
                idx = knn_point(self.nsample, xyz2, xyz1)
            
            neighbor_xyz = index_points_group(xyz2, idx)
            direction_xyz = (neighbor_xyz - xyz1.view(B, N1, 1, C)).permute(0, 3, 2, 1)

            grouped_points2_r = index_points_group(points2_r, idx).permute(0, 3, 2, 1)
            grouped_points2_r_o = index_points_group(points2_r_o, idx).permute(0, 3, 2, 1)
            grouped_points2_z = index_points_group(points2_z, idx).permute(0, 3, 2, 1)

        point1_graph = points1

        # r
        r = direction_xyz
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(points1)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1) + grouped_points2_r
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = direction_xyz
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(points1)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1) + grouped_points2_r_o
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        points1 = self.fuse_r_o(points1)
        points1_expand = points1.view(B, points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        points1_expand = r * points1_expand

        h = direction_xyz
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + points1_expand + grouped_points2_z
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        new_points = (1 - z) * points1 + z * h
     

        return new_points

#############################DIFFUSION###############################
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


import math
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
