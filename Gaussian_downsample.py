import numpy as np
import torch.nn.functional as F
import torch 

def gaussian_downsample(x, scale=1):

    assert scale in [1], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], 'reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale,padding=0)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(C, T, x.size(2), x.size(3))
    return x


if __name__ == "__main__":
    torch.cuda.set_device(3)
    input = torch.randn(1, 3, 320, 180).cuda()
    x = gaussian_downsample(input,4).cuda()
