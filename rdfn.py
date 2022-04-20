import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv
import functools

# ==========
# Spatio-temporal deformable fusion module
# ==========
class STDF(nn.Module):
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.middle_conv = nn.Sequential(
            nn.Conv2d(1, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.reconstructA = self.make_layer(functools.partial(ResBlock, nf, nf), 2)
        self.reconstructB = self.make_layer(functools.partial(ResBlock, nf, nf), 2)
        self.unet = NestedUNet()

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )
        self.out_TA = nn.Conv2d(6 * nf, nf, 1, 1,bias=True)
    def forward(self, inputs):

        in_nc = self.in_nc
        n_off_msk = self.size_dk

        middle = self.middle_conv(inputs[:, 3:4, :, :])
        neigbor = []
        feat_frame = []
        # feat_frame
        for j in range(7):
            if j != 3:
                a = torch.cat((inputs[:,3:4,:,:], inputs[:,j:j+1,:,:]), 1)
                neigbor.append(a)
                feat_frame.append(self.in_conv(a))
        Ht = []
        for j in range(len(feat_frame)):
            out_1 = self.unet(feat_frame[j])
            out_2 = self.unet(feat_frame[j] + out_1)
            off_msk = self.offset_mask(self.out_conv(out_1 + out_2))
            off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
            msk = torch.sigmoid(
                off_msk[:, in_nc * 2 * n_off_msk:, ...]
            )
            fused_feat = F.relu(
                self.deform_conv(neigbor[j], off, msk),
                inplace=True
            )

            e = middle-fused_feat
            e = self.reconstructA(e)
            middle = middle+e
            Ht.append(middle)
            middle = self.reconstructB(middle)

        out = torch.cat(Ht,1)
        out = self.out_TA(out)
        return out


# ==========
# Quality enhancement module
# ==========
class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()


        self.reconstruct = self.make_layer(functools.partial(PSABlock, nf), 7)
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, inputs):

        out = self.reconstruct(inputs)
        out = self.out_conv(out)
        return out




class PSABlock(nn.Module):
    def __init__(self, nf):
        super(PSABlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn3 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.psa = PSAModule(nf, nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.dcn1(self.lrelu(self.dcn0(x))))

        out = self.dcn3(self.lrelu(self.dcn2(x)))
        out = self.psa(out)
        return out + x







def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out


class NestedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        nf = 64
        base_ks = 3
        self.Down0_0 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.conv0_0 = ResBlock(64, 64)

        self.Down0_1 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.conv0_1 = ResBlock(64, 64)

        self.Down0_2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        self.conv0_2 = ResBlock(64, 64)

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv0_3 = ResBlock(128, 64)
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv0_4 = ResBlock(128, 64)
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # input torch.Size([1, 64, 540, 960])
        x0_0 = self.conv0_0(self.Down0_0(input))
        x0_1 = self.conv0_1(self.Down0_1(x0_0))
        # x0_1 torch.Size([1, 64, 135, 240])
        x0_2 = self.conv0_2(self.Down0_2(x0_1))
        up0_1 = self.Up1(x0_2)
        #         out = out[:,:,:,0:h,0:w].contiguous()
        b, _, h, w = x0_1.size()
        up0_1 = up0_1[:, :, 0:h, 0:w].contiguous()
        up0_2 = self.Up2(self.conv0_3(torch.cat([up0_1, x0_1], 1)))

        up0_3 = self.Up3(self.conv0_4(torch.cat([up0_2, x0_0], 1)))

        return up0_3






# ==========
# RDFN network
# ==========
class RDFN(nn.Module):
    def __init__(self):

        super(RDFN, self).__init__()
        self.radius = 3
        self.input_len = 2 * self.radius + 1
        nf=64
        self.in_nc = 1
        self.ffnet = STDF(
            in_nc=2,
            out_nc=nf,
            nf=nf,
            nb=3,
            deform_ks=3
        )
        self.qenet = PlainCNN(
            in_nc=nf,
            nf=nf,
            nb=16,
            out_nc=1
        )

    def forward(self, x):
        out = self.ffnet(x)
        out = self.qenet(out)
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # res: add middle frame
        return out



if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = RDFN().cuda()
    from thop import profile
    input = torch.randn(1, 7, 64, 64).cuda()

    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))



