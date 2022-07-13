import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    with torch.cuda.amp.autocast(enabled=False):
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
# "zeros" "border"


def warp_features(inp, flow, ):
    groups = flow.shape[1]//2  # NCHW
    samples = inp.shape[0]
    h = inp.shape[2]
    w = inp.shape[3]
    assert(flow.shape[0] == samples and flow.shape[2]
           == h and flow.shape[3] == w)
    chns = inp.shape[1]
    chns_per_group = chns // groups
    assert(flow.shape[1] % 2 == 0)
    assert(chns % groups == 0)
    inp = inp.contiguous().view(samples*groups, chns_per_group, h, w)
    flow = flow.contiguous().view(samples*groups, 2,  h, w)
    feat = warp(inp, flow)
    feat = feat.view(samples, chns, h, w)
    return feat


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)/2.0
    normalized_flow_map = np.concatenate(
        (flow_map_np[:, :, 0:1]/h/2.0, flow_map_np[:, :, 1:2]/w/2.0), axis=2)
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * \
        (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return (rgb_map.clip(0, 1)*255.0).astype(np.uint8)
