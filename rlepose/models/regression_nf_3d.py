# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
from easydict import EasyDict

from .builder import SPPE
from .layers.real_nvp import RealNVP
from .layers.Resnet import ResNet


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())
    # return nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))
    # return nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 2))


def nets3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3), nn.Tanh())
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())


def nett3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3))
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2))


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


@SPPE.register_module
class RegressFlow3D(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(RegressFlow3D, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048
        }[cfg['NUM_LAYERS']]

        self.root_idx = 0

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs, out_channel = self._make_fc_layer()

        self.fc_coord = Linear(out_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(out_channel, self.num_joints * 3)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        self.share_flow = True

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))
        # masks3d = torch.from_numpy(np.array([[1, 1, 0], [0, 0, 1]] * 3).astype(np.float32))

        self.flow2d = RealNVP(nets, nett, masks, prior)
        self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())

        return nn.Sequential(*fc_layers), input_channel

    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        # Positional Pooling
        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 3)
        assert out_coord.shape[2] == 3

        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 3)
        if not self.training:
            pred_jts[:, :, 2] = pred_jts[:, :, 2] - pred_jts[:, self.root_idx:self.root_idx + 1, 2]

        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        if labels is not None:
            gt_uvd = labels['target_uvd'].reshape(pred_jts.shape)
            gt_uvd_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
            gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

            assert pred_jts.shape == sigma.shape, (pred_jts.shape, sigma.shape)
            bar_mu = (pred_jts - gt_uvd) / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(BATCH_SIZE, self.num_joints, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )
        return output
