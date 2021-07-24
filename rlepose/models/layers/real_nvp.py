import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)
