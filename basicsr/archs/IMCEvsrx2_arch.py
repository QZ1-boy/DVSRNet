import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet
from .raftupnet_arch import Raft_upNet000  
from .raftupnet01_arch import Raft_upNet00, Raft_upNet,Raft_upNetx3,Raft_upNetx4
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init


class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)



################################    X2   ################################################




@ARCH_REGISTRY.register()
class ICMEVSRx2_FG_SPM(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper  IconVSRx2_FS_fg
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.scale = 2
        self.is_with_alignment = True

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)  #  SpyNet  Raft_upNet000

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat + 3, num_feat, num_block)

        self.flow_trunk = ConvResidualBlocks(2 * 3 + 3, 3, num_block)
        self.spam_trunk = ConvResidualBlocks(3+2 * num_feat, num_feat, num_block)

        # reconstruction
        self.conv_err = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_trans = nn.Conv2d(3*num_feat, num_feat, 3, 1, 1)
        self.conv_fetr = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat*4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.CSAM_module = CSAM_Module()
        self.deform_align = SecondOrderDeformableAlignment(
                    num_feat,
                    num_feat,
                    3,
                    padding=1,
                    deform_groups=2,  #  16
                    max_residue_magnitude=10)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # residual network
        self.in_conv = nn.Sequential(nn.Conv2d(3, num_feat, 3, padding=1), nn.ReLU(inplace=True))
        hid_conv_lst = []
        for _ in range(8 - 2):
            hid_conv_lst += [nn.Conv2d(num_feat, num_feat, 3, padding=1),nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(num_feat, 3, 3, padding=1)

        # define head module
        modules_head = [nn.Conv2d(3, num_feat, 3 ,1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1)]
        self.head = nn.Sequential(*modules_head)

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(3, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(num_feat//4, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feat // 4, 64, 3, 2, 1, output_padding=1),
        )
        self.tau = 1

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # print("x[:, i:i + num_frames].contiguous()",x[:, i:i + num_frames].shape)  # e([2, 7, 3, 128, 128])
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def FlowDeformableAlignment(self, feats, flow):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        nf, _, hf, wf = feats.shape
        n, t, h, w = flow.shape

        feat_prop = flow.new_zeros(n, 64, h, w)
        # print("[feat_prop 0 ]",feat_prop.shape)   #  [2, 64, 128, 128]
        feat_current = feats
        feat_current = feat_current.cuda()
        feat_prop = feat_prop.cuda()
        flow = flow.cuda()
        # print("cuda feat_current",feat_current.device)

        # second-order deformable alignment
        if self.is_with_alignment:
            # print("feat_prop feat_current",feat_prop.shape)
            flow_n1 = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  #  flow(0, 2, 3, 1)
            flow_n1 = flow_n1.cuda()
            # print("[flow_n1 0 ]",flow_n1.shape)   #  [4, 64, 128, 128]
            torch.cuda.empty_cache()
            cond = flow_warp(feat_current, flow.permute(0, 2, 3, 1))
            cond = cond.cuda()
            # print("[cond 0 ]",cond.shape)   #  ([2, 64, 128, 128])

            # flow-guided deformable convolution
            cond = torch.cat([cond, feat_current, flow_n1], dim=1)
            cond = cond.cuda()
            cond = self.conv_trans(cond)

            feat_prop = torch.cat([feat_prop, feats], dim=1)
            # feat_prop = self.conv_fetr(feat_prop)
            # print("[feats]",feats.shape)    #  ([2, 64, 128, 128])
            # print("[cond 1]",cond.shape)    #  ([2, 192, 128, 128])
            # print("[feat_prop]",feat_prop.shape)   #  [4, 128, 128, 128])
            torch.cuda.empty_cache()
            feat_prop = self.deform_align(feat_current, cond, flow)  # .cuda()  feat_prop
            feat_prop = feat_prop.cuda()
            torch.cuda.empty_cache()
            # print("[feat_prop]",feat_prop.shape)  #  ([4, 128, 128, 128])

        return feat_prop

    def forward(self, x):
        b, n, c, h_input, w_input = x.size()
        # x0 = x
        # x = F.interpolate(x, size=(c, self.scale*h_input, self.scale*w_input), mode='trilinear', align_corners=False)
        # x = self.pad_spatial(x)
        b, n, c, h, w = x.shape 

        # keyframe_idx = list(range(0, n, self.keyframe_stride))
        # if keyframe_idx[-1] != n - 1:
        #     keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        # feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        out_fw = []
        out_sm = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        flow = flows_backward.new_zeros(b, 2, h, w)
        spa_mask = flows_backward.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 backward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # spa_mask = spa_mask.view(b, n, -1, h, w)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
            out_fw.insert(0, flow)
            out_sm.insert(0, spa_mask)
            
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        flow = torch.zeros_like(flow)
        spa_mask = torch.zeros_like(spa_mask)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 forward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.forward_fusion(feat_prop)
            
            # flow_backprop = flow_warp(x_i, out_fw[i].permute(0, 2, 3, 1))
            # flow_foreprop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            
            # flow_prop = torch.cat([x_i, flow_backprop, flow_foreprop], dim=1)
            # flow_prop = self.flow_trunk(flow_prop)  #  joint optical flow fore - back
            # flow_prop = torch.cat([x_i, flow_prop], dim=1)  # 4 + 64
            # print("[out_sm[i]]",out_sm[i].shape)
            # print("[spa_mask]",spa_mask.shape)


            spam_prop = torch.cat([x_i, out_sm[i], spa_mask], dim=1)
            spam_prop = self.spam_trunk(spam_prop)  #  joint spa_mask  fore - back

            feat_prop = torch.cat([x_i, out_l[i], spam_prop, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            
            base = F.interpolate(x_i, scale_factor=2, mode='bilinear', align_corners=False)
           
            out_l[i] = out + base
        finout = torch.stack(out_l, dim=1)[..., :2 * h_input, :2 * w_input]

        return finout







@ARCH_REGISTRY.register()
class ICMEVSRx2_warp_SPM(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper  IconVSRx2_FS_fg
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.scale = 2
        self.is_with_alignment = True

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)  #  SpyNet Raft_upNet000

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat + 3, num_feat, num_block)

        self.flow_trunk = ConvResidualBlocks(2 * 3 + 3, 3, num_block)
        self.spam_trunk = ConvResidualBlocks(3+2 * num_feat, num_feat, num_block)

        # reconstruction
        self.conv_err = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_trans = nn.Conv2d(3*num_feat, num_feat, 3, 1, 1)
        self.conv_fetr = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat*4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.CSAM_module = CSAM_Module()
        self.deform_align = SecondOrderDeformableAlignment(
                    num_feat,
                    num_feat,
                    3,
                    padding=1,
                    deform_groups=2,  #  16
                    max_residue_magnitude=10)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # residual network
        self.in_conv = nn.Sequential(nn.Conv2d(3, num_feat, 3, padding=1), nn.ReLU(inplace=True))
        hid_conv_lst = []
        for _ in range(8 - 2):
            hid_conv_lst += [nn.Conv2d(num_feat, num_feat, 3, padding=1),nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(num_feat, 3, 3, padding=1)

        # define head module
        modules_head = [nn.Conv2d(3, num_feat, 3 ,1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1)]
        self.head = nn.Sequential(*modules_head)

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(3, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(num_feat//4, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feat // 4, 64, 3, 2, 1, output_padding=1),
        )
        self.tau = 1

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # print("x[:, i:i + num_frames].contiguous()",x[:, i:i + num_frames].shape)  # e([2, 7, 3, 128, 128])
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def FlowDeformableAlignment(self, feats, flow):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        nf, _, hf, wf = feats.shape
        n, t, h, w = flow.shape

        feat_prop = flow.new_zeros(n, 64, h, w)
        # print("[feat_prop 0 ]",feat_prop.shape)   #  [2, 64, 128, 128]
        feat_current = feats
        feat_current = feat_current.cuda()
        feat_prop = feat_prop.cuda()
        flow = flow.cuda()
        # print("cuda feat_current",feat_current.device)

        # second-order deformable alignment
        if self.is_with_alignment:
            # print("feat_prop feat_current",feat_prop.shape)
            flow_n1 = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  #  flow(0, 2, 3, 1)
            flow_n1 = flow_n1.cuda()
            # print("[flow_n1 0 ]",flow_n1.shape)   #  [4, 64, 128, 128]
            torch.cuda.empty_cache()
            cond = flow_warp(feat_current, flow.permute(0, 2, 3, 1))
            cond = cond.cuda()
            # print("[cond 0 ]",cond.shape)   #  ([2, 64, 128, 128])

            # flow-guided deformable convolution
            cond = torch.cat([cond, feat_current, flow_n1], dim=1)
            cond = cond.cuda()
            cond = self.conv_trans(cond)

            feat_prop = torch.cat([feat_prop, feats], dim=1)
            # feat_prop = self.conv_fetr(feat_prop)
            # print("[feats]",feats.shape)    #  ([2, 64, 128, 128])
            # print("[cond 1]",cond.shape)    #  ([2, 192, 128, 128])
            # print("[feat_prop]",feat_prop.shape)   #  [4, 128, 128, 128])
            torch.cuda.empty_cache()
            feat_prop = self.deform_align(feat_current, cond, flow)  # .cuda()  feat_prop
            feat_prop = feat_prop.cuda()
            torch.cuda.empty_cache()
            # print("[feat_prop]",feat_prop.shape)  #  ([4, 128, 128, 128])

        return feat_prop

    def forward(self, x):
        b, n, c, h_input, w_input = x.size()
        # x0 = x
        # x = F.interpolate(x, size=(c, self.scale*h_input, self.scale*w_input), mode='trilinear', align_corners=False)
        # x = self.pad_spatial(x)
        b, n, c, h, w = x.shape 

        # keyframe_idx = list(range(0, n, self.keyframe_stride))
        # if keyframe_idx[-1] != n - 1:
        #     keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        # feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        out_fw = []
        out_sm = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        flow = flows_backward.new_zeros(b, 2, h, w)
        spa_mask = flows_backward.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 backward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # spa_mask = spa_mask.view(b, n, -1, h, w)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
            out_fw.insert(0, flow)
            out_sm.insert(0, spa_mask)
            
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        flow = torch.zeros_like(flow)
        spa_mask = torch.zeros_like(spa_mask)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                # feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 forward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.forward_fusion(feat_prop)
            
            # flow_backprop = flow_warp(x_i, out_fw[i].permute(0, 2, 3, 1))
            # flow_foreprop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            
            # flow_prop = torch.cat([x_i, flow_backprop, flow_foreprop], dim=1)
            # flow_prop = self.flow_trunk(flow_prop)  #  joint optical flow fore - back
            # flow_prop = torch.cat([x_i, flow_prop], dim=1)  # 4 + 64
            # print("[out_sm[i]]",out_sm[i].shape)
            # print("[spa_mask]",spa_mask.shape)


            spam_prop = torch.cat([x_i, out_sm[i], spa_mask], dim=1)
            spam_prop = self.spam_trunk(spam_prop)  #  joint spa_mask  fore - back

            feat_prop = torch.cat([x_i, out_l[i], spam_prop, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            
            base = F.interpolate(x_i, scale_factor=2, mode='bilinear', align_corners=False)
           
            out_l[i] = out + base
        finout = torch.stack(out_l, dim=1)[..., :2 * h_input, :2 * w_input]

        return finout






@ARCH_REGISTRY.register()
class ICMEVSRx2_FG_only(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper  IconVSRx2_FS_fg
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.scale = 2
        self.is_with_alignment = True

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)  #  SpyNet  Raft_upNet000

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        self.flow_trunk = ConvResidualBlocks(2 * 3 + 3, 3, num_block)
        self.spam_trunk = ConvResidualBlocks(3+2 * num_feat, num_feat, num_block)

        # reconstruction
        self.conv_err = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_trans = nn.Conv2d(3*num_feat, num_feat, 3, 1, 1)
        self.conv_fetr = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat*4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.CSAM_module = CSAM_Module()
        self.deform_align = SecondOrderDeformableAlignment(
                    num_feat,
                    num_feat,
                    3,
                    padding=1,
                    deform_groups=2,  #  16
                    max_residue_magnitude=10)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # residual network
        self.in_conv = nn.Sequential(nn.Conv2d(3, num_feat, 3, padding=1), nn.ReLU(inplace=True))
        hid_conv_lst = []
        for _ in range(8 - 2):
            hid_conv_lst += [nn.Conv2d(num_feat, num_feat, 3, padding=1),nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(num_feat, 3, 3, padding=1)

        # define head module
        modules_head = [nn.Conv2d(3, num_feat, 3 ,1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1)]
        self.head = nn.Sequential(*modules_head)

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(3, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(num_feat//4, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feat // 4, 64, 3, 2, 1, output_padding=1),
        )
        self.tau = 1

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # print("x[:, i:i + num_frames].contiguous()",x[:, i:i + num_frames].shape)  # e([2, 7, 3, 128, 128])
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def FlowDeformableAlignment(self, feats, flow):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        nf, _, hf, wf = feats.shape
        n, t, h, w = flow.shape

        feat_prop = flow.new_zeros(n, 64, h, w)
        # print("[feat_prop 0 ]",feat_prop.shape)   #  [2, 64, 128, 128]
        feat_current = feats
        feat_current = feat_current.cuda()
        feat_prop = feat_prop.cuda()
        flow = flow.cuda()
        # print("cuda feat_current",feat_current.device)

        # second-order deformable alignment
        if self.is_with_alignment:
            # print("feat_prop feat_current",feat_prop.shape)
            flow_n1 = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  #  flow(0, 2, 3, 1)
            flow_n1 = flow_n1.cuda()
            # print("[flow_n1 0 ]",flow_n1.shape)   #  [4, 64, 128, 128]
            torch.cuda.empty_cache()
            cond = flow_warp(feat_current, flow.permute(0, 2, 3, 1))
            cond = cond.cuda()
            # print("[cond 0 ]",cond.shape)   #  ([2, 64, 128, 128])

            # flow-guided deformable convolution
            cond = torch.cat([cond, feat_current, flow_n1], dim=1)
            cond = cond.cuda()
            cond = self.conv_trans(cond)

            feat_prop = torch.cat([feat_prop, feats], dim=1)
            # feat_prop = self.conv_fetr(feat_prop)
            # print("[feats]",feats.shape)    #  ([2, 64, 128, 128])
            # print("[cond 1]",cond.shape)    #  ([2, 192, 128, 128])
            # print("[feat_prop]",feat_prop.shape)   #  [4, 128, 128, 128])
            torch.cuda.empty_cache()
            feat_prop = self.deform_align(feat_current, cond, flow)  # .cuda()  feat_prop
            feat_prop = feat_prop.cuda()
            torch.cuda.empty_cache()
            # print("[feat_prop]",feat_prop.shape)  #  ([4, 128, 128, 128])

        return feat_prop

    def forward(self, x):
        b, n, c, h_input, w_input = x.size()
        # x0 = x
        # x = F.interpolate(x, size=(c, self.scale*h_input, self.scale*w_input), mode='trilinear', align_corners=False)
        # x = self.pad_spatial(x)
        b, n, c, h, w = x.shape 

        # keyframe_idx = list(range(0, n, self.keyframe_stride))
        # if keyframe_idx[-1] != n - 1:
        #     keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        # feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        out_fw = []
        out_sm = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        flow = flows_backward.new_zeros(b, 2, h, w)
        spa_mask = flows_backward.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 backward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # spa_mask = spa_mask.view(b, n, -1, h, w)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
            out_fw.insert(0, flow)
            out_sm.insert(0, spa_mask)
            
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        flow = torch.zeros_like(flow)
        spa_mask = torch.zeros_like(spa_mask)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 forward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.forward_fusion(feat_prop)
            
            # flow_backprop = flow_warp(x_i, out_fw[i].permute(0, 2, 3, 1))
            # flow_foreprop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            
            # flow_prop = torch.cat([x_i, flow_backprop, flow_foreprop], dim=1)
            # flow_prop = self.flow_trunk(flow_prop)  #  joint optical flow fore - back
            # flow_prop = torch.cat([x_i, flow_prop], dim=1)  # 4 + 64
            # print("[out_sm[i]]",out_sm[i].shape)
            # print("[spa_mask]",spa_mask.shape)


            # spam_prop = torch.cat([x_i, out_sm[i], spa_mask], dim=1)
            # spam_prop = self.spam_trunk(spam_prop)  #  joint spa_mask  fore - back

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            
            base = F.interpolate(x_i, scale_factor=2, mode='bilinear', align_corners=False)
           
            out_l[i] = out + base
        finout = torch.stack(out_l, dim=1)[..., :2 * h_input, :2 * w_input]

        return finout



@ARCH_REGISTRY.register()
class ICMEVSRx2_warp_only(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper  IconVSRx2_FS_fg
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.scale = 2
        self.is_with_alignment = True

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)  #  SpyNet  Raft_upNet000

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        self.flow_trunk = ConvResidualBlocks(2 * 3 + 3, 3, num_block)
        self.spam_trunk = ConvResidualBlocks(3+2 * num_feat, num_feat, num_block)

        # reconstruction
        self.conv_err = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_trans = nn.Conv2d(3*num_feat, num_feat, 3, 1, 1)
        self.conv_fetr = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat*4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.CSAM_module = CSAM_Module()
        self.deform_align = SecondOrderDeformableAlignment(
                    num_feat,
                    num_feat,
                    3,
                    padding=1,
                    deform_groups=2,  #  16
                    max_residue_magnitude=10)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # residual network
        self.in_conv = nn.Sequential(nn.Conv2d(3, num_feat, 3, padding=1), nn.ReLU(inplace=True))
        hid_conv_lst = []
        for _ in range(8 - 2):
            hid_conv_lst += [nn.Conv2d(num_feat, num_feat, 3, padding=1),nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(num_feat, 3, 3, padding=1)

        # define head module
        modules_head = [nn.Conv2d(3, num_feat, 3 ,1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1)]
        self.head = nn.Sequential(*modules_head)

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(3, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(num_feat//4, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feat // 4, 64, 3, 2, 1, output_padding=1),
        )
        self.tau = 1

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # print("x[:, i:i + num_frames].contiguous()",x[:, i:i + num_frames].shape)  # e([2, 7, 3, 128, 128])
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def FlowDeformableAlignment(self, feats, flow):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        nf, _, hf, wf = feats.shape
        n, t, h, w = flow.shape

        feat_prop = flow.new_zeros(n, 64, h, w)
        # print("[feat_prop 0 ]",feat_prop.shape)   #  [2, 64, 128, 128]
        feat_current = feats
        feat_current = feat_current.cuda()
        feat_prop = feat_prop.cuda()
        flow = flow.cuda()
        # print("cuda feat_current",feat_current.device)

        # second-order deformable alignment
        if self.is_with_alignment:
            # print("feat_prop feat_current",feat_prop.shape)
            flow_n1 = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  #  flow(0, 2, 3, 1)
            flow_n1 = flow_n1.cuda()
            # print("[flow_n1 0 ]",flow_n1.shape)   #  [4, 64, 128, 128]
            torch.cuda.empty_cache()
            cond = flow_warp(feat_current, flow.permute(0, 2, 3, 1))
            cond = cond.cuda()
            # print("[cond 0 ]",cond.shape)   #  ([2, 64, 128, 128])

            # flow-guided deformable convolution
            cond = torch.cat([cond, feat_current, flow_n1], dim=1)
            cond = cond.cuda()
            cond = self.conv_trans(cond)

            feat_prop = torch.cat([feat_prop, feats], dim=1)
            # feat_prop = self.conv_fetr(feat_prop)
            # print("[feats]",feats.shape)    #  ([2, 64, 128, 128])
            # print("[cond 1]",cond.shape)    #  ([2, 192, 128, 128])
            # print("[feat_prop]",feat_prop.shape)   #  [4, 128, 128, 128])
            torch.cuda.empty_cache()
            feat_prop = self.deform_align(feat_current, cond, flow)  # .cuda()  feat_prop
            feat_prop = feat_prop.cuda()
            torch.cuda.empty_cache()
            # print("[feat_prop]",feat_prop.shape)  #  ([4, 128, 128, 128])

        return feat_prop

    def forward(self, x):
        b, n, c, h_input, w_input = x.size()
        # x0 = x
        # x = F.interpolate(x, size=(c, self.scale*h_input, self.scale*w_input), mode='trilinear', align_corners=False)
        # x = self.pad_spatial(x)
        b, n, c, h, w = x.shape 

        # keyframe_idx = list(range(0, n, self.keyframe_stride))
        # if keyframe_idx[-1] != n - 1:
        #     keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        # feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        out_fw = []
        out_sm = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        flow = flows_backward.new_zeros(b, 2, h, w)
        spa_mask = flows_backward.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 backward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # spa_mask = spa_mask.view(b, n, -1, h, w)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
            out_fw.insert(0, flow)
            out_sm.insert(0, spa_mask)
            
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        flow = torch.zeros_like(flow)
        spa_mask = torch.zeros_like(spa_mask)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                # feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 forward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                # feat_prop = self.forward_fusion(feat_prop)
            
            # flow_backprop = flow_warp(x_i, out_fw[i].permute(0, 2, 3, 1))
            # flow_foreprop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            
            # flow_prop = torch.cat([x_i, flow_backprop, flow_foreprop], dim=1)
            # flow_prop = self.flow_trunk(flow_prop)  #  joint optical flow fore - back
            # flow_prop = torch.cat([x_i, flow_prop], dim=1)  # 4 + 64
            # print("[out_sm[i]]",out_sm[i].shape)
            # print("[spa_mask]",spa_mask.shape)


            # spam_prop = torch.cat([x_i, out_sm[i], spa_mask], dim=1)
            # spam_prop = self.spam_trunk(spam_prop)  #  joint spa_mask  fore - back

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            
            base = F.interpolate(x_i, scale_factor=2, mode='bilinear', align_corners=False)
           
            out_l[i] = out + base
        finout = torch.stack(out_l, dim=1)[..., :2 * h_input, :2 * w_input]

        return finout






@ARCH_REGISTRY.register()
class ICMEVSRx2_FG_FB_SPM(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper  IconVSRx2_FS_fg
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.scale = 2
        self.is_with_alignment = True

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)  #  SpyNet Raft_upNet000

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat + 3 + 3, num_feat, num_block)

        self.flow_trunk = ConvResidualBlocks(2 * 3 + 3, 3, num_block)
        self.spam_trunk = ConvResidualBlocks(3+2 * num_feat, num_feat, num_block)

        # reconstruction
        self.conv_err = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_trans = nn.Conv2d(3*num_feat, num_feat, 3, 1, 1)
        self.conv_fetr = nn.Conv2d(2*num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.CSAM_module = CSAM_Module()
        self.deform_align = SecondOrderDeformableAlignment(
                    num_feat,
                    num_feat,
                    3,
                    padding=1,
                    deform_groups=2,  #  16
                    max_residue_magnitude=10)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # residual network
        self.in_conv = nn.Sequential(nn.Conv2d(3, num_feat, 3, padding=1), nn.ReLU(inplace=True))
        hid_conv_lst = []
        for _ in range(8 - 2):
            hid_conv_lst += [nn.Conv2d(num_feat, num_feat, 3, padding=1),nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(num_feat, 3, 3, padding=1)

        # define head module
        modules_head = [nn.Conv2d(3, num_feat, 3 ,1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1)]
        self.head = nn.Sequential(*modules_head)

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(3, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(num_feat//4, num_feat//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feat // 4, 64, 3, 2, 1, output_padding=1),
        )
        self.tau = 1

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, self.scale*h, self.scale*w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, self.scale*h, self.scale*w)

        return flows_forward, flows_backward

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # print("x[:, i:i + num_frames].contiguous()",x[:, i:i + num_frames].shape)  # e([2, 7, 3, 128, 128])
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def FlowDeformableAlignment(self, feats, flow):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        nf, _, hf, wf = feats.shape
        n, t, h, w = flow.shape

        feat_prop = flow.new_zeros(n, 64, h, w)
        # print("[feat_prop 0 ]",feat_prop.shape)   #  [2, 64, 128, 128]
        feat_current = feats
        feat_current = feat_current.cuda()
        feat_prop = feat_prop.cuda()
        flow = flow.cuda()
        # print("cuda feat_current",feat_current.device)

        # second-order deformable alignment
        if self.is_with_alignment:
            # print("feat_prop feat_current",feat_prop.shape)
            flow_n1 = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  #  flow(0, 2, 3, 1)
            flow_n1 = flow_n1.cuda()
            # print("[flow_n1 0 ]",flow_n1.shape)   #  [4, 64, 128, 128]
            torch.cuda.empty_cache()
            cond = flow_warp(feat_current, flow.permute(0, 2, 3, 1))
            cond = cond.cuda()
            # print("[cond 0 ]",cond.shape)   #  ([2, 64, 128, 128])

            # flow-guided deformable convolution
            cond = torch.cat([cond, feat_current, flow_n1], dim=1)
            cond = cond.cuda()
            cond = self.conv_trans(cond)

            feat_prop = torch.cat([feat_prop, feats], dim=1)
            # feat_prop = self.conv_fetr(feat_prop)
            # print("[feats]",feats.shape)    #  ([2, 64, 128, 128])
            # print("[cond 1]",cond.shape)    #  ([2, 192, 128, 128])
            # print("[feat_prop]",feat_prop.shape)   #  [4, 128, 128, 128])
            torch.cuda.empty_cache()
            feat_prop = self.deform_align(feat_current, cond, flow)  # .cuda()  feat_prop
            feat_prop = feat_prop.cuda()
            torch.cuda.empty_cache()
            # print("[feat_prop]",feat_prop.shape)  #  ([4, 128, 128, 128])

        return feat_prop

    def forward(self, x):
        b, n, c, h_input, w_input = x.size()
        x0 = x
        x = F.interpolate(x, size=(c, self.scale*h_input, self.scale*w_input), mode='trilinear', align_corners=False)
        # x = self.pad_spatial(x)
        b, n, c, h, w = x.shape 

        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x0)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        out_fw = []
        out_sm = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        flow = flows_backward.new_zeros(b, 2, h, w)
        spa_mask = flows_backward.new_zeros(b, 3, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
            if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 backward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                # spa_mask = spa_mask.view(b, n, -1, h, w)
                feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
            out_fw.insert(0, flow)
            out_sm.insert(0, spa_mask)
            
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        flow = torch.zeros_like(flow)
        spa_mask = torch.zeros_like(spa_mask)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = self.FlowDeformableAlignment(feat_prop, flow)
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                x_mk = x_i.view(-1, c, h, w)  # self.head(x.view(-1, c, h, w))
                # print("[x_mk]",x_mk.shape)   # ([2, 3, 128, 128])
                spa_mask = self.spa_mask(x_mk)
                # print("[spa_mask 1 forward]",i)   #  [28, 3, 128, 128])
                spa_mask = self.gumbel_softmax(spa_mask, 1, self.tau)
                feat_prop = torch.cat([feat_prop,  feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)
            
            flow_backprop = flow_warp(x_i, out_fw[i].permute(0, 2, 3, 1))
            flow_foreprop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            
            flow_prop = torch.cat([x_i, flow_backprop, flow_foreprop], dim=1)
            flow_prop = self.flow_trunk(flow_prop)  #  joint optical flow fore - back
            # flow_prop = torch.cat([x_i, flow_prop], dim=1)  # 4 + 64

            spam_prop = torch.cat([x_i, out_sm[i], spa_mask], dim=1)
            spam_prop = self.spam_trunk(spam_prop)  #  joint spa_mask  fore - back

            feat_prop = torch.cat([x_i, out_l[i], flow_prop, spam_prop, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.upconv1(feat_prop))
            # out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            # out_res = self.in_conv(out)
            # out_res = self.hid_conv(out_res)
            # residual_out = self.out_conv(out_res)
            # # print("[residual_out new3]",residual_out.shape)   #  ([2, 128, 128, 128])

            # #(2) residual fusion  
            # # residual leaning
            # err = x_i - out #  + down_res
            # out_err = self.lrelu(self.conv_err(err))
            # out_err = out_err + residual_out
            # resid_out = self.conv_err(out_err) #  + x_i  # self.lrelu

            # out = out + x_i + resid_out
            out_l[i] = out + x_i
        finout = torch.stack(out_l, dim=1)[..., :2 * h_input, :2 * w_input]

        return finout




class EDVRFeatureExtractor(nn.Module):

    def __init__(self, num_input_frame, num_feat, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extrat pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels+2 , self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        # print('[x]',x.shape)  # ([2, 128, 128, 128])
        # print('[extra_feat]',extra_feat.shape)  #  [2, 192, 128, 128])
        # print('[flow_1]',flow_1.shape)  #  ([2, 2, 128, 128])

        extra_feat = torch.cat([extra_feat, flow_1], dim=1)   #  conncat 的内容修改 offset的获得是由相邻帧的信息得到 不需要太多
        # extra_feat = flow_warp(extra_feat, flow_1)
        # print('[extra_feat]',extra_feat.shape)  # [4, 196, 128, 128]
        torch.cuda.empty_cache()
        out = self.conv_offset(extra_feat)   #  卷积 非线性、卷积 非线性
        # print('[out]',out.shape)  # [4, 216, 128, 128]  ([4, 108, 128, 128])
        torch.cuda.empty_cache()
        # o1, mask = torch.chunk(out, 2, dim=1)  # 分解成两个tensor ，一个当作偏移用，一个当作mask用 
        o1, o2 , mask = torch.chunk(out, 3, dim=1)
        # print('[o1]',o1.shape)  #  [4, 72, 128, 128]
        # print('[o2]',o2.shape)  #  [4, 72, 128, 128]

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))     #torch.tanh(torch.cat((o1, o2), dim=1))
        # print('[offset]',offset.shape)  # [4, 144, 128, 128]
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        offset = offset + flow_1.flip(1).repeat(1,offset.size(1) // 2, 1, 1)   #  两个偏移叠加 
        # print('[offset 1]',offset.shape)  # [4, 144, 128, 128]k

        # mask  进行修改和改进 mask的优化 
        mask = torch.sigmoid(mask)  #  非线性激活
        # print('[mask]',mask.shape)   # ([4, 144, 128, 128])
        # print('[x]',x.shape)  #  ([4, 128, 128, 128])
        # print('[offset]',offset.shape)  #  ([4, 144, 128, 128])
        # print('[mask]',mask.shape)  #  ([4, 72, 128, 128])
        # x = x[:,:64,:,:].contiguous()
        # offset = offset[:,21,:,:].contiguous()  #   offset 与 mask ：必须是两倍的mask
        # mask = mask[:,:36,:,:].contiguous()

        torch.cuda.empty_cache()
        OUT = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
        # print("[OUT]",OUT.shape)   # ([1, 64, 128, 128])

        return OUT




class SecondOrderDeformableAlignment02(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels+2 , self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        # print('[x]',x.shape)  # ([2, 128, 128, 128])
        # print('[extra_feat]',extra_feat.shape)  #  [2, 192, 128, 128])
        # print('[flow_1]',flow_1.shape)  #  ([2, 2, 128, 128])

        extra_feat = torch.cat([extra_feat, flow_1], dim=1)   #  conncat 的内容修改 offset的获得是由相邻帧的信息得到 不需要太多
        # extra_feat = flow_warp(extra_feat, flow_1)
        # print('[extra_feat]',extra_feat.shape)  # [4, 196, 128, 128]
        torch.cuda.empty_cache()
        out = self.conv_offset(extra_feat)   #  卷积 非线性、卷积 非线性
        # print('[out]',out.shape)  # [4, 216, 128, 128]  ([4, 108, 128, 128])
        torch.cuda.empty_cache()
        # o1, mask = torch.chunk(out, 2, dim=1)  # 分解成两个tensor ，一个当作偏移用，一个当作mask用 
        o1, o2 , mask = torch.chunk(out, 3, dim=1)
        # print('[o1]',o1.shape)  #  [4, 72, 128, 128]
        # print('[o2]',o2.shape)  #  [4, 72, 128, 128]

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))     #torch.tanh(torch.cat((o1, o2), dim=1))
        # print('[offset]',offset.shape)  # [4, 144, 128, 128]
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        offset = offset + flow_1.flip(1).repeat(1,offset.size(1) // 2, 1, 1)   #  两个偏移叠加 
        # print('[offset 1]',offset.shape)  # [4, 144, 128, 128]k

        # mask  进行修改和改进 mask的优化 
        mask = torch.sigmoid(mask)  #  非线性激活
        # print('[mask]',mask.shape)   # ([4, 144, 128, 128])
        # print('[x]',x.shape)  #  ([4, 128, 128, 128])
        # print('[offset]',offset.shape)  #  ([4, 144, 128, 128])
        # print('[mask]',mask.shape)  #  ([4, 72, 128, 128])
        # x = x[:,:64,:,:].contiguous()
        # offset = offset[:,21,:,:].contiguous()  #   offset 与 mask ：必须是两倍的mask
        # mask = mask[:,:36,:,:].contiguous()

        torch.cuda.empty_cache()
        OUT = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
        # print("[OUT]",OUT.shape)   # ([1, 64, 128, 128])

        return OUT






class SecondOrderDeformableAlignment01(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment01, self).__init__(*args, **kwargs)

        # self.conv_mask = nn.Conv2d(64, 27 * self.deform_groups//2, 3, 1, 1)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64 , self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups//2, 3, 1, 1),)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels+2 , self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        # print('[x]',x.shape)  # ([2, 128, 128, 128])
        # print('[extra_feat]',extra_feat.shape)  #  [2, 192, 128, 128])
        # print('[flow_1]',flow_1.shape)  #  ([2, 2, 128, 128])

        extra_feat = torch.cat([extra_feat, flow_1], dim=1)   #  conncat 的内容修改 offset的获得是由相邻帧的信息得到 不需要flow
        # extra_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        # print('[extra_feat]',extra_feat.shape)  # [4, 196, 128, 128]
        torch.cuda.empty_cache()
        out = self.conv_offset(extra_feat)   #  卷积 非线性、卷积 非线性
        # print('[out]',out.shape)  # [4, 216, 128, 128]  ([4, 108, 128, 128])
        torch.cuda.empty_cache()
        # o1, mask = torch.chunk(out, 2, dim=1)  # 分解成两个tensor ，一个当作偏移用，一个当作mask用 
        # o1, o2 , mask = torch.chunk(out, 3, dim=1)
        # print('[o1]',o1.shape)  #  [4, 72, 128, 128]
        # print('[o2]',o2.shape)  #  [4, 72, 128, 128]

        # offset
        offset = self.max_residue_magnitude * out     

        # offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))     #torch.tanh(torch.cat((o1, o2), dim=1))
        # print('[offset]',offset.shape)  # [4, 144, 128, 128]
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        offset = offset + flow_1.flip(1).repeat(1,offset.size(1) // 2, 1, 1)   #  两个偏移叠加 
        # print('[offset 1]',offset.shape)  # [4, 144, 128, 128]k

        # mask  进行修改和改进 mask的优化 
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)  #  非线性激活
        # print('[mask]',mask.shape)   # ([4, 144, 128, 128])
        # print('[x]',x.shape)  #  ([4, 128, 128, 128])
        # print('[offset]',offset.shape)  #  ([4, 144, 128, 128])
        # print('[mask]',mask.shape)  #  ([4, 72, 128, 128])
        # x = x[:,:64,:,:].contiguous()
        # offset = offset[:,21,:,:].contiguous()  #   offset 与 mask ：必须是两倍的mask
        # mask = mask[:,:36,:,:].contiguous()

        torch.cuda.empty_cache()
        OUT = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
        # print("[OUT]",OUT.shape)   # ([1, 64, 128, 128])

        return OUT



class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self):
        super(CSAM_Module, self).__init__()
        # self.chanel_in = in_dim
        self.conv = nn.Conv3d(7, 1, 3, 1, 1)
        self.conv_x = nn.Conv2d(128, 3, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B  N  C  H  W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        # print('[CSAM_Module before shape]',x2.shape)
        # x = x2.permute(0,2,3,1)
        x = x2  # .unsqueeze(1)
        print('[CSAM_Module before shape 1]',x.shape)

        m_batchsize, N, C, height, width = x.size()
        # print('[CSAM_Module shape]',x.shape)
        # out = x.squeeze(1)
        out = x1
        out = self.sigmoid(self.conv(out))
        
        proj_query = x.contiguous().view(m_batchsize, N, -1)
        proj_key = x.contiguous().view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.contiguous().view(m_batchsize, N, C, height, width)

        out = self.gamma*out   #  去掉
        out = out.contiguous().view(m_batchsize,N, -1, height, width)
        # print('[CSAM_Module x1 shape]',x1.shape)   #  [2, 3, 128, 128]
        # print('[CSAM_Module out shape]',out.shape)  #  [2, 128, 128, 128]
        # x = x * out + x
        # xo = self.conv_x(x2 * out) + x1
        print('[CSAM_Module x1 shape]',x1.shape)  #  ([60, 1, 2, 128, 128])
        print('[CSAM_Module out shape]',out.shape)  #  ([4, 15, 3, 128, 128])
        # print('[CSAM_Module after shape]',xo.shape)
        xo = x1 * out + x1

        # xo = xo.permute(0,3,1,2)
        # print('[CSAM_Module after shape]',xo.shape)

        return xo



class Backward_warp(nn.Module):

    def __init__(self):
        super(Backward_warp,self).__init__()


    def _meshgrid(self,height,width):

        y_t = torch.linspace(0,height - 1, height).reshape(height,1) * torch.ones(1,width)
        x_t = torch.ones(height,1) * torch.linspace(0, width - 1, width).reshape(1,width)

        x_t_flat = x_t.reshape(1,1,height,width)
        y_t_flat = y_t.reshape(1,1,height,width)

        grid = torch.cat((x_t_flat,y_t_flat),1)

        return grid


    def _interpolate(self,img , x, y , out_height, out_width):

        num_batch,height,width,num_channel = img.size()
        height_f = float(height)
        width_f = float(width)

        x = torch.clamp(x,0,width - 1)
        y = torch.clamp(y,0,height - 1)

        x0_f = x.floor().cuda()
        y0_f = y.floor().cuda()
        x1_f = x0_f + 1.0
        y1_f = y0_f + 1.0

        x0 = torch.tensor(x0_f, dtype = torch.int64)
        y0 = torch.tensor(y0_f, dtype = torch.int64)
        x1 = torch.tensor(torch.clamp(x1_f, 0, width_f -1), dtype = torch.int64)
        y1 = torch.tensor(torch.clamp(y1_f, 0, height_f -1), dtype = torch.int64)
 
        dim1 = width * height
        dim2 = width
        base = torch.tensor((torch.arange(num_batch) * dim1),dtype = torch.int64).cuda()
        base = base.reshape(num_batch,1).repeat(1,out_height * out_width).view(-1).cuda()

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.reshape(-1,num_channel)

        Ia = img_flat[idx_a]
        Ib = img_flat[idx_b]
        Ic = img_flat[idx_c]
        Id = img_flat[idx_d]

        wa = ((x1_f-x) * (y1_f-y)).reshape(-1,1)
        wb = ((x1_f-x) * (y-y0_f)).reshape(-1,1)
        wc = ((x-x0_f) * (y1_f-y)).reshape(-1,1)
        wd = ((x-x0_f) * (y-y0_f)).reshape(-1,1)
        output = wa * Ia + wb * Ib + wc * Ic + wd *Id

        return output


    def _transform_flow(self,flow,input,downsample_factor):

        num_batch,num_channel,height,width = input.size()

        out_height = height
        out_width = width
        grid = self._meshgrid(height, width)
        if num_batch > 1:
            grid = grid.repeat(num_batch,1,1,1)

        control_point = grid.cuda() + flow
        input_t = input.permute(0,2,3,1)

        x_s_flat = control_point[:,0,:,:].contiguous().view(-1)
        y_s_flat = control_point[:,1,:,:].contiguous().view(-1)

        input_transformed = self._interpolate(input_t,x_s_flat,y_s_flat,out_height,out_width)

        input_transformed = input_transformed.reshape(num_batch,out_height,out_width,num_channel)

        output = input_transformed.permute(0,3,1,2)

        return output

    def forward(self,input,flow,downsample_factor = 1):

        return self._transform_flow(flow,input, downsample_factor)