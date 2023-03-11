import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


class ResBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.skip = skip if skip else nn.Identity()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConv2d(ResBlock):
    def __init__(self, c_in, c_mid, c_out):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        ], skip)


class SelfAtt2d(nn.Module):
    def __init__(self, c_in, n_head=1):
        super().__init__()
        assert c_in % n_head == 0
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(input)
        qkv = qkv.view(
            [n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        out = input + self.out_proj(y)
        return out


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.skip(input), self.main(input)], dim=1)


class MsgPass(nn.Module):
    def __init__(self, c_in, cmut_size, heads=1, chns_div=4):
        super().__init__()
        self.cmut_size = cmut_size
        self.chns_div = chns_div
        self.gate_proj = nn.Conv2d(c_in, heads, 1)

    def forward(self, input):
        weight = self.gate_proj(input)
        cmut_size = self.cmut_size[0]

        n, c, h, w = input.shape
        n2, c2, h2, w2 = weight.shape
        assert (h2 == h and w2 == w)
        input = input.reshape(n//cmut_size, cmut_size, c, h, w)
        weight = weight.reshape(n2//cmut_size, cmut_size, c2, h2, w2)

        c_msg = c//self.chns_div
        input_msg = input[:, :, :c_msg, :, :]
        input_other = input[:, :, c_msg:, :, :]
        weight = torch.softmax(weight, dim=1)
        input_msg = torch.sum(input_msg*weight, dim=1, keepdim=True).repeat(
            [1,  cmut_size, 1, 1, 1])
        output = torch.cat([input_msg, input_other], dim=2)
        return output.reshape(n, c, h, w)


class CINN(nn.Module):
    def __init__(self, inp_chns, out_chns, c=64+32):
        super().__init__()

        self.cmut_size = [-1]
        cmut_size = self.cmut_size
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.net = nn.Sequential(
            ResConv2d(inp_chns, cs[0], cs[0]),
            ResConv2d(cs[0], cs[0], cs[0]),
            ResConv2d(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,
                ResConv2d(cs[0], cs[1], cs[1]),
                ResConv2d(cs[1], cs[1], cs[1]),
                ResConv2d(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ResConv2d(cs[1], cs[2], cs[2]),
                    ResConv2d(cs[2], cs[2], cs[2]),
                    ResConv2d(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ResConv2d(cs[2], cs[3], cs[3]),
                        ResConv2d(cs[3], cs[3], cs[3]),
                        ResConv2d(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ResConv2d(cs[3], cs[4], cs[4]),
                            ResConv2d(cs[4], cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ResConv2d(cs[4], cs[5], cs[5]),
                                MsgPass(cs[5], cmut_size),
                                SelfAtt2d(cs[5], cs[5] // 128),
                                ResConv2d(cs[5], cs[5], cs[5]),
                                MsgPass(cs[5], cmut_size),
                                SelfAtt2d(cs[5], cs[5] // 128),
                                ResConv2d(cs[5], cs[5], cs[5]),
                                MsgPass(cs[5], cmut_size),
                                SelfAtt2d(cs[5], cs[5] // 128),
                                ResConv2d(cs[5], cs[5], cs[4]),
                                MsgPass(cs[4], cmut_size),
                                SelfAtt2d(cs[4], cs[4] // 128),
                                self.up,
                            ]),
                            ResConv2d(cs[4] + cs[4], cs[4], cs[4]*3),
                            MsgPass(cs[3]*3, cmut_size),
                            ResConv2d(cs[4]*3, cs[4], cs[4]),
                            SelfAtt2d(cs[4], cs[4] // 128),
                            ResConv2d(cs[4], cs[4], cs[3]*3),
                            MsgPass(cs[3]*3, cmut_size),
                            ResConv2d(cs[3]*3, cs[3], cs[3]),
                            SelfAtt2d(cs[3], cs[3] // 128),
                            self.up,
                        ]),
                        ResConv2d(cs[3] + cs[3], cs[3], cs[3]*3),
                        MsgPass(cs[3]*3, cmut_size),
                        ResConv2d(cs[3]*3, cs[3], cs[3]),
                        SelfAtt2d(cs[3], cs[3] // 128),
                        ResConv2d(cs[3], cs[3], cs[2]*3),
                        MsgPass(cs[2]*3, cmut_size),
                        ResConv2d(cs[2]*3, cs[2], cs[2]),
                        SelfAtt2d(cs[2], cs[2] // 128),
                        self.up,
                    ]),
                    ResConv2d(cs[2] + cs[2], cs[2], cs[2]*3),
                    DefConv(cs[2]*3, cs[2]//16),
                    ResConv2d(cs[2]*3, cs[2], cs[2]*3),
                    MsgPass(cs[2]*3, cmut_size),
                    ResConv2d(cs[2]*3, cs[2], cs[1]),
                    self.up,
                ]),
                ResConv2d(cs[1] + cs[1], cs[1], cs[1]*3),
                DefConv(cs[1]*3, cs[1]//16),
                ResConv2d(cs[1]*3, cs[1], cs[1]*3),
                MsgPass(cs[1]*3, cmut_size),
                ResConv2d(cs[1]*3, cs[1], cs[0]),
                self.up,
            ]),
            ResConv2d(cs[0] + cs[0], cs[0], cs[0]*3),
            DefConv(cs[0] * 3, cs[0]//16),
            ResConv2d(cs[0]*3, cs[0], out_chns),
        )

    def forward(self, target, input):
        self.cmut_size[0] = int(input.shape[1])
        if target is not None:
            interm = torch.unsqueeze(target, 1).repeat(
                [1,  input.shape[1], 1, 1, 1]).view(input.shape[0], input.shape[1], target.shape[1], input.shape[3], input.shape[4])
            inp = torch.cat([input, interm], dim=2)
        else:
            inp = input
        inp = inp.reshape(
            input.shape[0]*input.shape[1], -1, input.shape[3], input.shape[4])
        inp = self.net(inp)
        return inp.reshape(input.shape[0],  input.shape[1], -1, input.shape[3], input.shape[4])


class DefConv(nn.Module):
    def __init__(self,  c_in, n_head=1):
        super().__init__()
        self.c_in = c_in
        self.n_head = n_head
        self.flow_proj = nn.Sequential(
            nn.Conv2d(c_in, c_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, n_head * 2, 1)
        )

    def warp(self, tenInput, tenFlow):
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            backwarp_tenGrid[k] = torch.cat(
                [tenHorizontal, tenVertical], 1).to(device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
        if tenInput.dtype != g.dtype:
            g = g.to(tenInput.dtype)
        return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, inp):
        flow = self.flow_proj(inp)
        heads = flow.shape[1]//2
        assert heads == self.n_head  # NCHW
        n, c, h, w = inp.shape
        assert(flow.shape[0] == n and flow.shape[2]
               == h and flow.shape[3] == w)

        chns_per_group = c // heads
        assert(flow.shape[1] % 2 == 0)
        assert(c % heads == 0)
        inp = inp.contiguous().view(n*heads, chns_per_group, h, w)
        flow = flow.contiguous().view(n*heads, 2,  h, w)
        feat = self.warp(inp, flow)
        feat = feat.view(n, c, h, w)
        return feat
