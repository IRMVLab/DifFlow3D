
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, CrossLayerLightFeatCosine as CrossLayer, FlowEmbeddingLayer, BidirectionalLayerFeatCosine
from pointconv_util import SceneFlowEstimatorResidual
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance, knn_point_cosine, knn_point
#from pointconv_util import DiffusionFlowResidual
from pointconv_util import exists, default, extract, cosine_beta_schedule, SinusoidalPosEmb
from pointconv_util import index_points_gather
import time
from tqdm.auto import tqdm

scale = 1.0


class PointConvEncoder(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32

        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet = weightnet) # out
        self.level0_1 = Conv1d(32, 64)
        
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet = weightnet)
        self.level1_0 = Conv1d(64, 64)# out
        self.level1_1 = Conv1d(64, 128)

        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet = weightnet)
        self.level2_0 = Conv1d(128, 128) # out
        self.level2_1 = Conv1d(128, 256)

        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet = weightnet)
        self.level3_0 = Conv1d(256, 256) # out
        self.level3_1 = Conv1d(256, 512)

        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256, weightnet = weightnet) # out

    def forward(self, xyz, color):
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        #l1
        pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        #l2
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)

        #l3
        pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)

        #l4
        pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)

        return [xyz, pc_l1, pc_l2, pc_l3, pc_l4], \
                [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4], \
                [fps_l1, fps_l2, fps_l3, fps_l4]

class RecurrentUnit(nn.Module):
    def __init__(self, iters, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2, 
                weightnet=8, flow_channels = [64, 64], flow_mlp = [64, 64]):
        super(RecurrentUnit, self).__init__()
        flow_nei = 32
        self.iters = iters
        self.scale = scale
        
        self.bid = BidirectionalLayerFeatCosine(flow_nei, feat_new_ch+feat_ch, cross_mlp1)
        self.fe = FlowEmbeddingLayer(flow_nei, cross_mlp1[-1], cross_mlp2)
        #self.flow = SceneFlowGRUResidual(latent_ch, cross_mlp2[-1] + feat_ch, channels = flow_channels, mlp = flow_mlp)
        # add
        neighbors = 9
        self.flow = DiffusionSceneFlowGRUResidual(neighbors, in_channel=cross_mlp2[-1]+feat_ch, latent_channel=latent_ch, mlp=flow_channels, channels=flow_channels)

        self.warping = PointWarping()

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow = None,certainty = None, uncertainty = 0.5):
        
        c_feat1 = torch.cat([feat1, feat1_new], dim = 1)
        c_feat2 = torch.cat([feat2, feat2_new], dim = 1)

        flows = []
        for i in range(self.iters):
            pc2_warp = self.warping(pc1, pc2, up_flow)
            feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1, feat2)
            fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1, feat2)
            new_feat1 = torch.cat([feat1, fe], dim = 1)
            #feat_flow, flow = self.flow(pc1, up_feat, new_feat1, up_flow, gt_flow)
            if self.training:
                feat_flow, flow, certainty_new, loss = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow,certainty, uncertainty)
            else:
                feat_flow, flow, certainty_new = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow,certainty, uncertainty)
            up_flow = flow
            up_feat = feat_flow
            c_feat1 = torch.cat([feat1, feat1_new], dim = 1)
            c_feat2 = torch.cat([feat2, feat2_new], dim = 1)
            flows.append(flow)
        if self.training:
            return flows, feat1_new, feat2_new, feat_flow, certainty_new,loss
        else:
            return flows, feat1_new, feat2_new, feat_flow, certainty_new


class GRUMappingNoGCN(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = False, use_leaky = True, return_inter=False, radius=None, use_relu=False):
        super(GRUMappingNoGCN,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu

        last_channel = in_channel + 3

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2,flow = None, flow_gt = None):
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
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = points1

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
            else:
                r = self.relu(r)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
            else:
                z = self.relu(z)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.use_relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        new_points = (1 - z) * points1 + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class DiffusionSceneFlowGRUResidual(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = False, use_leaky = True, \
                 return_inter=False, radius=None, use_relu=False, channels = [64,64], clamp = [-200,200],scale_dif = 1.0):
        super(DiffusionSceneFlowGRUResidual,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu
        self.fc = nn.Conv1d(channels[-1],4,1)
        self.clamp = clamp

        #last_channel = in_channel + 3
        last_channel = in_channel + 3 + 64 + 3 + 1

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

        # build diffusion
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        # define beta schedule
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = scale_dif
        # time embeddings
        time_dim = 64
        dim = 16
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)



        self.iters = 1
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def forward(self, xyz1, xyz2, points1, points2, flow, flow_gt, certainty, uncertainty=0.5):
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
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)
        
        # add
        batch_size = flow.shape[0]
        n = flow.shape[2]

        if self.training:
            flow_gt = flow_gt.permute(0,2,1)
            certainty = certainty.permute(0,2,1)

            gt_certainty_norm = torch.norm(flow_gt - flow, dim=-2)
            sf_norm = torch.norm(flow_gt, dim=-2)
            relative_err = gt_certainty_norm / (sf_norm + 1e-4)

            def z_score_normalize(data):
                mean = np.mean(data, axis=1)
                std_dev = np.std(data, axis=1)
                normalized_data = (data - mean) / std_dev
                return normalized_data

            #gt_certainty = z_score_normalize(gt_certainty_norm)
            gt_certainty = torch.where(torch.logical_or(gt_certainty_norm < uncertainty, relative_err < uncertainty), torch.ones_like(gt_certainty_norm), torch.zeros_like(gt_certainty_norm))


            gt_certainty = torch.unsqueeze(gt_certainty, dim=2)

            gt_delta_certainty = gt_certainty - certainty
            gt_delta_certainty = gt_delta_certainty.detach()
            gt_delta_flow = flow_gt - flow
            gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow)
            gt_delta_flow = gt_delta_flow.detach()

            t = torch.randint(0, self.timesteps, (batch_size,), device= flow.device).long()
            noise = (self.scale * torch.randn_like(gt_delta_flow)).float()
            noise_certainty = (self.scale * torch.randn_like(gt_delta_certainty)).float()

            delta_flow = self.q_sample(x_start=gt_delta_flow, t=t, noise=noise)
            flow_new = flow + delta_flow
            delta_certainty = self.q_sample(x_start=gt_delta_certainty, t=t, noise=noise_certainty)
            certainty_new = certainty + delta_certainty

            for i in range(self.iters):
                delta_flow = delta_flow.detach()
                flow_new = flow_new.detach()
                time = self.time_mlp(t)
                delta_certainty = delta_certainty.detach()
                certainty_new = certainty_new.detach()
                # #print(time.shape)
                time = time.unsqueeze(1).repeat(1, n, 1)
  
                if self.radius is None:
                    sqrdists = square_distance(xyz1, xyz2)
                    dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
                    neighbor_xyz = index_points_group(xyz2, knn_idx)
                    direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                    grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
                    time = time.unsqueeze(-2).repeat(1,1,self.nsample,1)
                    delta_flow = delta_flow.permute(0,2,1)
                    delta_flow = delta_flow.unsqueeze(-2).repeat(1,1,self.nsample,1)
                    delta_certainty = delta_certainty.unsqueeze(-2).repeat(1,1,self.nsample,1)

                    new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty, delta_flow, time], dim = -1) # B, N1, nsample, D1+D2+3

                    new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

                else:
                    new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                    new_points = new_points.permute(0, 1, 3, 2)

                point1_graph = points1

                # r
                r = new_points
                for i, conv in enumerate(self.mlp_r_convs):
                    r = conv(r)
                    if i == 0:
                        grouped_points1 = self.fuse_r(point1_graph)
                        r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        r = self.mlp_r_bns[i](r)
                    if i == len(self.mlp_r_convs) - 1:
                        r = self.sigmoid(r)
                    else:
                        r = self.relu(r)


                # z
                z = new_points
                for i, conv in enumerate(self.mlp_z_convs):
                    z = conv(z)
                    if i == 0:
                        grouped_points1 = self.fuse_z(point1_graph)
                        z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        z = self.mlp_z_bns[i](z)
                    if i == len(self.mlp_z_convs) - 1:
                        z = self.sigmoid(z)
                    else:
                        z = self.relu(z)

                    if i == len(self.mlp_z_convs) - 2:
                        z = torch.max(z, -2)[0].unsqueeze(-2)
                
                z = z.squeeze(-2)

                point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                point1_expand = r * point1_graph_expand
                point1_expand = self.fuse_r_o(point1_expand)

                h = new_points
                for i, conv in enumerate(self.mlp_h_convs):
                    h = conv(h)
                    if i == 0:
                        h = h + point1_expand
                    if self.bn:
                        h = self.mlp_h_bns[i](h)
                    if i == len(self.mlp_h_convs) - 1:
                        # 
                        if self.use_relu:
                            h = self.relu(h)
                        else:
                            h = self.tanh(h)
                    else:
                        h = self.relu(h)
                    if i == len(self.mlp_h_convs) - 2:
                        h = torch.max(h, -2)[0].unsqueeze(-2)

                h = h.squeeze(-2)

                new_points = (1 - z) * points1 + z * h

                if self.mlp2:
                    for _, conv in enumerate(self.mlp2):
                        new_points = conv(new_points)        

                new_points_delta = new_points - points1
                update = self.fc(new_points_delta)
                delta_flow, delta_certainty = update[:, :3, :].clamp(self.clamp[0], self.clamp[1]), update[:, 3:, :]
                certainty = certainty.permute(0,2,1)
                certainty_new = certainty + delta_certainty
                if flow is None:
                    flow = delta_flow
                else:
                    flow_new = delta_flow + flow
                
                loss_df = F.mse_loss(delta_flow, gt_delta_flow)
                gt_delta_certainty = gt_delta_certainty.permute(0,2,1)
                loss_dc = F.mse_loss(delta_certainty, gt_delta_certainty)
                loss = loss_df + loss_dc
            return new_points, flow_new, certainty_new, loss
        else:
            batch, device, total_timesteps, sampling_timesteps, eta = batch_size, flow.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            img = (self.scale * torch.randn_like(flow)).float()
            img_certainty = (self.scale * torch.randn_like(certainty)).float()

            #for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            for time, time_next in time_pairs:
                t = torch.full((batch,), time, device=device, dtype=torch.long)

                delta_flow = img
                flow_new = flow + delta_flow

                delta_certainty = img_certainty
                certainty_new = certainty + delta_certainty

                for i in range(self.iters):
                    delta_flow = delta_flow.detach()
                    flow_new = flow_new.detach()

                    delta_certainty = delta_certainty.detach()
                    certainty_new = certainty_new.detach()

                    time = self.time_mlp(t)
                    # #print(time.shape)
                    time = time.unsqueeze(1).repeat(1, n, 1)
    
                    if self.radius is None:
                        sqrdists = square_distance(xyz1, xyz2)
                        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
                        neighbor_xyz = index_points_group(xyz2, knn_idx)
                        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                        grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
                        time = time.unsqueeze(-2).repeat(1,1,self.nsample,1)
                        delta_flow = delta_flow.permute(0,2,1)
                        delta_flow = delta_flow.unsqueeze(-2).repeat(1,1,self.nsample,1)
                        delta_certainty = delta_certainty.permute(0,2,1)
                        delta_certainty = delta_certainty.unsqueeze(-2).repeat(1,1,self.nsample,1)


                        #points1_1 = points1.permute(0, 2, 1)
                        #points1_1 = points1_1.unsqueeze(-2).repeat(1, 1, self.nsample, 1)

                        new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty, delta_flow, time], dim = -1) # B, N1, nsample, D1+D2+3
                        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

                    else:
                        new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                        new_points = new_points.permute(0, 1, 3, 2)

                    point1_graph = points1

                    # r
                    r = new_points
                    for i, conv in enumerate(self.mlp_r_convs):
                        r = conv(r)
                        if i == 0:
                            grouped_points1 = self.fuse_r(point1_graph)
                            r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                        if self.bn:
                            r = self.mlp_r_bns[i](r)
                        if i == len(self.mlp_r_convs) - 1:
                            r = self.sigmoid(r)
                        else:
                            r = self.relu(r)


                    # z
                    z = new_points
                    for i, conv in enumerate(self.mlp_z_convs):
                        z = conv(z)
                        if i == 0:
                            grouped_points1 = self.fuse_z(point1_graph)
                            z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                        if self.bn:
                            z = self.mlp_z_bns[i](z)
                        if i == len(self.mlp_z_convs) - 1:
                            z = self.sigmoid(z)
                            # #print('sigmoid', z.shape)
                        else:
                            z = self.relu(z)
                            # #print('relu', z.shape)

                        if i == len(self.mlp_z_convs) - 2:
                            z = torch.max(z, -2)[0].unsqueeze(-2)
                            # #print('max', z.shape)
                    
                    z = z.squeeze(-2)

                    point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    point1_expand = r * point1_graph_expand
                    point1_expand = self.fuse_r_o(point1_expand)

                    h = new_points
                    for i, conv in enumerate(self.mlp_h_convs):
                        h = conv(h)
                        if i == 0:
                            h = h + point1_expand
                        if self.bn:
                            h = self.mlp_h_bns[i](h)
                        if i == len(self.mlp_h_convs) - 1:
                            # 
                            if self.use_relu:
                                h = self.relu(h)
                            else:
                                h = self.tanh(h)
                        else:
                            h = self.relu(h)
                        if i == len(self.mlp_h_convs) - 2:
                            h = torch.max(h, -2)[0].unsqueeze(-2)

                    h = h.squeeze(-2)

                    new_points = (1 - z) * points1 + z * h

                    if self.mlp2:
                        for _, conv in enumerate(self.mlp2):
                            new_points = conv(new_points)        
                    
                    new_points_delta = new_points - points1
                    #delta_flow = self.fc(new_points_delta).clamp(self.clamp[0], self.clamp[1]) 
                    update = self.fc(new_points_delta)
                    delta_flow, delta_certainty = update[:, :3, :].clamp(self.clamp[0], self.clamp[1]), update[:, 3:, :]
                    
                    certainty_new = certainty + delta_certainty

                    if flow is None:
                        flow_new = delta_flow
                    else:
                        flow_new = delta_flow + flow


                pred_noise = self.predict_noise_from_start(img, t, delta_flow)
                pred_noise_certainty = self.predict_noise_from_start(img_certainty, t, delta_certainty)

                if time_next < 0:
                    delta_flow = delta_flow
                    delta_certainty = delta_certainty
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(flow)).float()
                noise_certainty = (self.snr_scale * torch.randn_like(certainty)).float()

                img = delta_flow * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
                img_certainty = delta_certainty * alpha_next.sqrt() + c * pred_noise_certainty + sigma * noise_certainty
            
            return new_points, flow_new, certainty_new


class SceneFlowGRUResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [64, 64], mlp = [64, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowGRUResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        # last_channel = feat_ch + cost_ch

        self.gru = GRUMappingNoGCN(neighbors, in_channel=cost_ch, latent_channel=feat_ch, mlp=channels)
        
        # self.mlp_convs = nn.ModuleList()
        # for _, ch_out in enumerate(mlp):
        #     self.mlp_convs.append(Conv1d(last_channel, ch_out))
        #     last_channel = ch_out

        self.fc = nn.Conv1d(channels[-1], 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None, flow_gt = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        # new_points = torch.cat([feats, cost_volume], dim = 1)

        feats_new = self.gru(xyz, xyz, feats, cost_volume,flow,flow_gt)

        new_points = feats_new-feats
        # for conv in self.mlp_convs:
        #     new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return feats_new, flow

class PointConvBidirection(nn.Module):
    def __init__(self, iters=3):
        super(PointConvBidirection, self).__init__()
        flow_nei = 32
        weightnet = 8
        self.scale = scale
        self.iters = iters

        self.encoder = PointConvEncoder(weightnet=weightnet)

        #l0: 8192
        self.recurrent0 = RecurrentUnit(iters=iters, feat_ch=32, feat_new_ch=32, latent_ch=64, cross_mlp1=[32, 32], cross_mlp2=[32, 32], weightnet=weightnet, flow_channels = [64, 64], flow_mlp = [64, 64])

        #l1: 2048
        self.recurrent1 = RecurrentUnit(iters=iters, feat_ch=64, feat_new_ch=64, latent_ch=64, cross_mlp1=[64, 64], cross_mlp2=[64, 64], weightnet=weightnet)

        #l2: 512
        self.recurrent2 = RecurrentUnit(iters=iters, feat_ch=128, feat_new_ch=128, latent_ch=64, cross_mlp1=[128, 128], cross_mlp2=[128, 128], weightnet=weightnet)

        #l3: 256
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256, 256, channels = [128, 64], mlp=[], weightnet = weightnet)
        #self.flow_dif = DiffusionFlowResidual(256, 256, channels = [128, 64], mlp=[], weightnet = weightnet)

        #deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2, gt_flow, uncertainty = 0.5):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N

        pc1s, feat1s, idx1s = self.encoder(xyz1, color1)
        pc2s, feat2s, idx2s = self.encoder(xyz2, color2)

        #l4
        feat1_l4_3 = self.upsample(pc1s[3], pc1s[4], feat1s[4])
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)
        feat2_l4_3 = self.upsample(pc2s[3], pc2s[4], feat2s[4])
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)
        
        # add
        # gt_flow
        l2_label = index_points_gather(gt_flow, idx1s[1])
        l1_label = index_points_gather(gt_flow, idx1s[0])
        l0_label = gt_flow

        #l3
        c_feat1_l3 = torch.cat([feat1s[3], feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2s[3], feat2_l4_3], dim = 1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1s[3], pc2s[3], c_feat1_l3, c_feat2_l3, feat1s[3], feat2s[3])
        feat3, flow3, certainty3 = self.flow3(pc1s[3], feat1s[3], cross3)
        #print(f'certainty3: {certainty3.shape}')
        #certainty3 = certainty3.permute(0,2,1)
        #feat3, flow3 = self.flow_dif(pc1s[3], feat1s[3], cross3, gt_flow, idx1s[2])

        feat1_l3_2 = self.upsample(pc1s[2], pc1s[3], feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2s[2], pc2s[3], feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        #l2
        up_flow2 = self.upsample(pc1s[2], pc1s[3], self.scale * flow3)
        up_certainty2 = self.upsample(pc1s[2], pc1s[3], self.scale * certainty3)
        #print(f'up_certainty2_1: {up_certainty2.shape}')
        up_feat2 = self.upsample(pc1s[2], pc1s[3], feat3)
        
        #flows2, feat1_new_l2, feat2_new_l2, feat2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2], up_flow2, up_feat2)

        if self.training:
            l2_flow_gt = l2_label
            flows2, feat1_new_l2, feat2_new_l2, feat2 ,certainty2,loss_l2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2], up_flow2, up_feat2, l2_flow_gt,up_certainty2, uncertainty)
        else:
            l2_flow_gt = None
            flows2, feat1_new_l2, feat2_new_l2, feat2 ,certainty2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2], up_flow2, up_feat2, l2_flow_gt,up_certainty2, uncertainty)

        feat1_l2_1 = self.upsample(pc1s[1], pc1s[2], feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)
        feat2_l2_1 = self.upsample(pc2s[1], pc2s[2], feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        #l1
        up_flow1 = self.upsample(pc1s[1], pc1s[2], self.scale * flows2[-1])
        #print(f'certainty2: {certainty2.shape}')
        up_certainty1 = self.upsample(pc1s[1], pc1s[2], self.scale * certainty2)
        #print(f'up_certainty1: {up_certainty1.shape}')
        up_feat1 = self.upsample(pc1s[1], pc1s[2], feat2)

        if self.training:
            l1_flow_gt = l1_label
            flows1, feat1_new_l1, feat2_new_l1, feat1, certainty1,loss_l1 = self.recurrent1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1], feat2s[1], up_flow1, up_feat1, l1_flow_gt,up_certainty1, uncertainty)
        else:
            l1_flow_gt = None
            flows1, feat1_new_l1, feat2_new_l1, feat1, certainty1 = self.recurrent1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1], feat2s[1], up_flow1, up_feat1, l1_flow_gt,up_certainty1, uncertainty)

        feat1_l1_0 = self.upsample(pc1s[0], pc1s[1], feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)
        feat2_l1_0 = self.upsample(pc2s[0], pc2s[1], feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        #l0
        up_flow0 = self.upsample(pc1s[0], pc1s[1], self.scale * flows1[-1])
        up_certainty0 = self.upsample(pc1s[0], pc1s[1], self.scale * certainty1)
        up_feat0 = self.upsample(pc1s[0], pc1s[1], feat1)

        if self.training:
            l0_gt_flow = l0_label
            flows0, feat1_new_l0, feat2_new_l0, feat0, certainty0,loss_l0= self.recurrent0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0], feat2s[0], up_flow0, up_feat0, l0_gt_flow, up_certainty0, uncertainty)
        else:
            l0_gt_flow = None
            flows0, feat1_new_l0, feat2_new_l0, feat0 ,certainty0= self.recurrent0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0], feat2s[0], up_flow0, up_feat0, l0_gt_flow, up_certainty0, uncertainty)

        # flows = np.concatenate((flows0[::-1], flows1[::-1], flows2[::-1], [flow3]))
        flows = [flows0[::-1], flows1[::-1], flows2[::-1], [flow3]]
        pc1 = pc1s
        pc2 = pc2s
        fps_pc1_idxs = [[None for _ in range(self.iters-1)], [idx1s[0]], [idx1s[1]], [idx1s[2]]]
        fps_pc2_idxs = [[None for _ in range(self.iters-1)], [idx2s[0]], [idx2s[1]], [idx2s[2]]]

        if self.training:
            return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, loss_l2, loss_l1, loss_l0
        else:
            #print(f'self.iters:{self.iters}')
            #print(f'fps:{fps_pc1_idxs}')
            return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2

def multiScaleLoss(pred_flows, gt_flow, fps_idxs, loss_f2 = None,loss_f1 = None,loss_f0 = None, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    #generate GT list and mask1s
    gt_flows = [gt_flow]
    alphas = [alpha[0]]
    a = 0
    
    for i in range(1, len(fps_idxs)+1):
        #print(f'fps_idx:{fps_idxs[i-1]}')
        fps_idx = fps_idxs[i - 1][0]
        if fps_idx is not None:
            sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
            gt_flows.append(sub_gt_flow)
            a += 1
            alphas.append(alpha[a])
        else:
            # gt_flows.append(gt_flows[-1])
            alphas.append(alpha[a])

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        # import pdb
        # pdb.set_trace()
        diff_flow = pred_flows[i][0].permute(0, 2, 1) - gt_flows[i]
        total_loss += alphas[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
        if loss_f2 is not None:
          total_loss = total_loss + alpha[2]*loss_f2.mean()
        if loss_f1 is not None:
          total_loss = total_loss + alpha[1]*loss_f1.mean()
        if loss_f0 is not None:
          total_loss = total_loss + alpha[0]*loss_f0.mean()

    return total_loss

def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 32, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 31.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows, fps_idxs, iters):
    f_curvature = 0.3
    f_smoothness = 4.0
    f_chamfer = 1.0
    f_distill = 0.1

    #num of scale
    num_scale = len(pred_flows) - iters -1

    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    distillation_loss = torch.zeros(1).cuda()
    l = 0
    for i in range(num_scale):
        cur_flow = pred_flows[i] # B 3 N
        if i == 0 or (i > 0 and fps_idxs[i-1] is not None):
            cur_pc1 = pc1[l] # B 3 N
            cur_pc2 = pc2[l]
            l += 1
        #compute curvature
        cur_pc2_curvature = curvature(cur_pc2)
        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()


        # if i == 0:
        #     flow_distil = cur_flow.detach()
        #     selfDistillLoss = 0
        # if i > 0 and fps_idxs[i-1] is not None:
        #     flow_distil = index_points(flow_distil.transpose(1,2), fps_idxs[i-1]).transpose(1,2)
        # if i > 0:
        #     selfDistillLoss = torch.norm(cur_flow - flow_distil, dim = 1).sum(dim = 1).mean()
        
        #smoothness
        # smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[l-1] * chamferLoss
        if l < 2:
            smoothness_loss += alpha[l-1] * computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        curvature_loss += alpha[l-1] * curvatureLoss
        # distillation_loss += alpha[l-1] * selfDistillLoss
    total_loss = f_chamfer * chamfer_loss+ f_smoothness * smoothness_loss + f_curvature * curvature_loss  # + f_distill * distillation_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss, distillation_loss


from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    input = torch.randn((1,8192,3)).float().cuda()
    model = PointConvBidirection(iters=1).cuda()
    # #print(model)
    output = model(input,input,input,input)
    macs, params = profile(model, inputs=(input,input,input,input))
    macs, params = clever_format([macs, params], "%.3f")
    #print(macs, params)
    total = sum([param.nelement() for param in model.parameters()])
    #print("Number of parameter: %.2fM" % (total/1e6))

    from ptflops import get_model_complexity_info
    def prepare_input(resolution):
        x1 = torch.FloatTensor(1,8192,3)
        return [x1,x1,x1,x1]

    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=True)
    #print('      - Flops:  ' + flops)
    #print('      - Params: ' + params)
    # for n,p in model.named_parameters():
    #     #print(p.numel(), "\t", n, p.shape, )
    # dump_input = torch.randn((1,8192,3)).float().cuda()
    # traced_model = torch.jit.trace(model, (dump_input, dump_input, dump_input, dump_input))

    # timer = 0
    # for i in range(100):
    #     t = time.time()
    #     _ = traced_model(input,input,input,input)
    #     timer += time.time() - t
    # #print(timer / 100.0)
