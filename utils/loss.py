import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, out, gt):
        grad_out_x = out[:, :, :, 1:] - out[:, :, :, :-1]
        grad_out_y = out[:, :, 1:, :] - out[:, :, :-1, :]

        grad_gt_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        grad_gt_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        loss_x = self.l1(grad_out_x, grad_gt_x)
        loss_y = self.l1(grad_out_y, grad_gt_y)

        loss = self.weight * (loss_x + loss_y)

        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss




class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = EdgeLoss().cuda()

    input_1 = torch.randn(16, 16).cuda()
    input_2 = torch.randn(16, 16).cuda()
    loss = net(input_1,input_2)
    print(0.05*loss)

