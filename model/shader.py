import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp_features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, c=224, out_msgs=0, out_locals=0, block_nums=1, out_masks=1, out_local_flows=32, out_msgs_flows=32, out_feat_flows=0):

        super(DecoderBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c, 3, 2, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 2, 1),
            nn.PReLU(c),
        )

        self.convblocks = nn.ModuleList()
        for i in range(block_nums):
            self.convblocks.append(nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.PReLU(c),
            ))
        self.out_flows = 2
        self.out_msgs = out_msgs
        self.out_msgs_flows = out_msgs_flows if out_msgs > 0 else 0
        self.out_locals = out_locals
        self.out_local_flows = out_local_flows if out_locals > 0 else 0
        self.out_masks = out_masks
        self.out_feat_flows = out_feat_flows

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, 2, 1),
            nn.PReLU(c),
            nn.ConvTranspose2d(c, self.out_flows+self.out_msgs+self.out_msgs_flows +
                               self.out_locals+self.out_local_flows+self.out_masks+self.out_feat_flows, 4, 2, 1),
        )

    def forward(self, accumulated_flow, *other):
        x = [accumulated_flow]
        for each in other:
            if each is not None:
                assert(accumulated_flow.shape[-1] == each.shape[-1]), "decoder want {}, but get {}".format(
                    accumulated_flow.shape, each.shape)
                x.append(each)
        feat = self.conv0(torch.cat(x, dim=1))
        for convblock1 in self.convblocks:
            feat = convblock1(feat) + feat
        feat = self.conv_last(feat)
        prev = 0
        flow = feat[:, prev:prev+self.out_flows, :, :]
        prev += self.out_flows
        message = feat[:, prev:prev+self.out_msgs,
                       :, :] if self.out_msgs > 0 else None
        prev += self.out_msgs
        message_flow = feat[:, prev:prev + self.out_msgs_flows,
                            :, :] if self.out_msgs_flows > 0 else None
        prev += self.out_msgs_flows
        local_message = feat[:, prev:prev + self.out_locals,
                             :, :] if self.out_locals > 0 else None
        prev += self.out_locals
        local_message_flow = feat[:, prev:prev+self.out_local_flows,
                                  :, :] if self.out_local_flows > 0 else None
        prev += self.out_local_flows
        mask = torch.sigmoid(
            feat[:, prev:prev+self.out_masks, :, :]) if self.out_masks > 0 else None
        prev += self.out_masks
        feat_flow = feat[:, prev:prev+self.out_feat_flows,
                         :, :] if self.out_feat_flows > 0 else None
        prev += self.out_feat_flows
        return flow, mask, message, message_flow, local_message, local_message_flow, feat_flow


class CINN(nn.Module):
    def __init__(self, DIM_SHADER_REFERENCE, target_feature_chns=[512, 256, 128, 64, 64], feature_chns=[2048, 1024, 512, 256, 64], out_msgs_chn=[2048, 1024, 512, 256, 64, 64], out_locals_chn=[2048, 1024, 512, 256, 64, 0], block_num=[1, 1, 1, 1, 1, 2], block_chn_num=[224, 224, 224, 224, 224, 224]):
        super(CINN, self).__init__()
       
        self.in_msgs_chn = [0, *out_msgs_chn[:-1]]
        self.in_locals_chn = [0, *out_locals_chn[:-1]]

        self.decoder_blocks = nn.ModuleList()
        self.feed_weighted = True
        if self.feed_weighted:
            in_planes = 2+2+DIM_SHADER_REFERENCE*2
        else:
            in_planes = 2+DIM_SHADER_REFERENCE
        for each_target_feature_chns, each_feature_chns, each_out_msgs_chn, each_out_locals_chn, each_in_msgs_chn, each_in_locals_chn, each_block_num, each_block_chn_num in zip(target_feature_chns, feature_chns, out_msgs_chn, out_locals_chn, self.in_msgs_chn, self.in_locals_chn, block_num, block_chn_num):
            self.decoder_blocks.append(
                DecoderBlock(in_planes+each_target_feature_chns+each_feature_chns+each_in_locals_chn+each_in_msgs_chn, c=each_block_chn_num, block_nums=each_block_num, out_msgs=each_out_msgs_chn, out_locals=each_out_locals_chn, out_masks=2+each_out_locals_chn))
        for i in range(len(feature_chns), len(out_locals_chn)):
            #print("append extra block", i, "msg",
            #      out_msgs_chn[i], "local", out_locals_chn[i], "block", block_num[i])
            self.decoder_blocks.append(
                DecoderBlock(in_planes+self.in_msgs_chn[i]+self.in_locals_chn[i], c=block_chn_num[i], block_nums=block_num[i], out_msgs=out_msgs_chn[i], out_locals=out_locals_chn[i], out_masks=2+out_msgs_chn[i], out_feat_flows=0))

    def apply_flow(self, mask, message, message_flow, local_message, local_message_flow, x_reference, accumulated_flow, each_x_reference_features=None, each_x_reference_features_flow=None):
        if each_x_reference_features is not None:
            size_from = each_x_reference_features
        else:
            size_from = x_reference
        f_size = (size_from.shape[2], size_from.shape[3])
        accumulated_flow = self.flow_rescale(
            accumulated_flow, size_from)
        # mask = warp_features(F.interpolate(
        #    mask, size=f_size, mode="bilinear"), accumulated_flow) if mask is not None else None
        mask = F.interpolate(
            mask, size=f_size, mode="bilinear") if mask is not None else None
        message = F.interpolate(
            message, size=f_size, mode="bilinear") if message is not None else None
        message_flow = self.flow_rescale(
            message_flow, size_from) if message_flow is not None else None
        message = warp_features(
            message, message_flow) if message_flow is not None else message

        local_message = F.interpolate(
            local_message, size=f_size, mode="bilinear") if local_message is not None else None
        local_message_flow = self.flow_rescale(
            local_message_flow, size_from) if local_message_flow is not None else None
        local_message = warp_features(
            local_message, local_message_flow) if local_message_flow is not None else local_message

        warp_x_reference = warp_features(F.interpolate(
            x_reference, size=f_size, mode="bilinear"), accumulated_flow)

        each_x_reference_features_flow = self.flow_rescale(
            each_x_reference_features_flow, size_from) if (each_x_reference_features is not None and each_x_reference_features_flow is not None) else None
        warp_each_x_reference_features = warp_features(
            each_x_reference_features, each_x_reference_features_flow) if each_x_reference_features_flow is not None else each_x_reference_features

        return mask, message, local_message, warp_x_reference, accumulated_flow, warp_each_x_reference_features, each_x_reference_features_flow

    def forward(self, x_target_features=[], x_reference=None, x_reference_features=[]):
        y_flow = []
        y_feat_flow = []

        y_local_message = []
        y_warp_x_reference = []
        y_warp_x_reference_features = []

        y_weighted_flow = []
        y_weighted_mask = []
        y_weighted_message = []
        y_weighted_x_reference = []
        y_weighted_x_reference_features = []

        for pyrlevel, ifblock in enumerate(self.decoder_blocks):
            stacked_wref = []
            stacked_feat = []
            stacked_anci = []
            stacked_flow = []
            stacked_mask = []
            stacked_mesg = []
            stacked_locm = []
            stacked_feat_flow = []
            for view_id in range(x_reference.shape[1]):  # NMCHW

                if pyrlevel == 0:
                    # create from zero flow
                    feat_ev = x_reference_features[pyrlevel][:,
                                                             view_id, :, :, :] if pyrlevel < len(x_reference_features) else None

                    accumulated_flow = torch.zeros_like(
                        feat_ev[:, :2, :, :]).to(device)
                    accumulated_feat_flow = torch.zeros_like(
                        feat_ev[:, :32, :, :]).to(device)
                    # domestic inputs
                    warp_x_reference = F.interpolate(x_reference[:, view_id, :, :, :], size=(
                        feat_ev.shape[-2], feat_ev.shape[-1]), mode="bilinear")
                    warp_x_reference_features = feat_ev

                    local_message = None
                    # federated inputs
                    weighted_flow = accumulated_flow if self.feed_weighted else None
                    weighted_wref = warp_x_reference if self.feed_weighted else None
                    weighted_message = None
                else:
                    # resume from last layer
                    accumulated_flow = y_flow[-1][:, view_id, :, :, :]
                    accumulated_feat_flow = y_feat_flow[-1][:,
                                                            view_id, :, :, :] if y_feat_flow[-1] is not None else None
                    # domestic inputs
                    warp_x_reference = y_warp_x_reference[-1][:,
                                                              view_id, :, :, :]
                    warp_x_reference_features = y_warp_x_reference_features[-1][:,
                                                                                view_id, :, :, :] if y_warp_x_reference_features[-1] is not None else None
                    local_message = y_local_message[-1][:, view_id, :,
                                                        :, :] if len(y_local_message) > 0 else None

                    # federated inputs
                    weighted_flow = y_weighted_flow[-1] if self.feed_weighted else None
                    weighted_wref = y_weighted_x_reference[-1] if self.feed_weighted else None
                    weighted_message = y_weighted_message[-1] if len(
                        y_weighted_message) > 0 else None
                scaled_x_target = x_target_features[pyrlevel][:, :, :, :].detach() if pyrlevel < len(
                    x_target_features) else None
                # compute flow
                residual_flow, mask, message, message_flow, local_message, local_message_flow, residual_feat_flow = ifblock(
                    accumulated_flow, scaled_x_target, warp_x_reference, warp_x_reference_features, weighted_flow, weighted_wref, weighted_message, local_message)
                accumulated_flow = residual_flow + accumulated_flow
                accumulated_feat_flow = accumulated_flow

                feat_ev = x_reference_features[pyrlevel+1][:,
                                                           view_id, :, :, :] if pyrlevel+1 < len(x_reference_features) else None
                mask, message, local_message, warp_x_reference, accumulated_flow,  warp_x_reference_features, accumulated_feat_flow = self.apply_flow(
                    mask, message, message_flow, local_message, local_message_flow, x_reference[:, view_id, :, :, :], accumulated_flow, feat_ev, accumulated_feat_flow)
                stacked_flow.append(accumulated_flow)
                if accumulated_feat_flow is not None:
                    stacked_feat_flow.append(accumulated_feat_flow)
                stacked_mask.append(mask)
                if message is not None:
                    stacked_mesg.append(message)
                if local_message is not None:
                    stacked_locm.append(local_message)
                stacked_wref.append(warp_x_reference)
                if warp_x_reference_features is not None:
                    stacked_feat.append(warp_x_reference_features)

            stacked_flow = torch.stack(stacked_flow, dim=1)  # M*NCHW -> NMCHW
            stacked_feat_flow = torch.stack(stacked_feat_flow, dim=1) if len(
                stacked_feat_flow) > 0 else None
            stacked_mask = torch.stack(
                stacked_mask, dim=1)
            
            stacked_mesg = torch.stack(stacked_mesg, dim=1) if len(
                stacked_mesg) > 0 else None
            stacked_locm = torch.stack(stacked_locm, dim=1) if len(
                stacked_locm) > 0 else None

            stacked_wref = torch.stack(stacked_wref, dim=1)
            stacked_feat = torch.stack(stacked_feat, dim=1) if len(
                stacked_feat) > 0 else None
            stacked_anci = torch.stack(stacked_anci, dim=1) if len(
                stacked_anci) > 0 else None
            y_flow.append(stacked_flow)
            y_feat_flow.append(stacked_feat_flow)

            y_warp_x_reference.append(stacked_wref)
            y_warp_x_reference_features.append(stacked_feat)
            # compute normalized confidence
            stacked_contrib = torch.nn.functional.softmax(stacked_mask, dim=1)

            # torch.sum to remove temp dimension M from NMCHW --> NCHW
            weighted_flow = torch.sum(
                stacked_mask[:, :, 0:1, :, :] * stacked_contrib[:, :, 0:1, :, :] * stacked_flow, dim=1)
            weighted_mask = torch.sum(
                stacked_contrib[:, :, 0:1, :, :] * stacked_mask[:, :, 0:1, :, :], dim=1)
            weighted_wref = torch.sum(
                stacked_mask[:, :, 0:1, :, :] * stacked_contrib[:, :, 0:1, :, :] * stacked_wref, dim=1) if stacked_wref is not None else None
            weighted_feat = torch.sum(
                stacked_mask[:, :, 1:2, :, :] * stacked_contrib[:, :, 1:2, :, :] * stacked_feat, dim=1) if stacked_feat is not None else None
            weighted_mesg = torch.sum(
                stacked_mask[:, :, 2:, :, :] * stacked_contrib[:, :, 2:, :, :] * stacked_mesg, dim=1) if stacked_mesg is not None else None
            y_weighted_flow.append(weighted_flow)
            y_weighted_mask.append(weighted_mask)
            if weighted_mesg is not None:
                y_weighted_message.append(weighted_mesg)
            if stacked_locm is not None:
                y_local_message.append(stacked_locm)
            y_weighted_message.append(weighted_mesg)
            y_weighted_x_reference.append(weighted_wref)
            y_weighted_x_reference_features.append(weighted_feat)

            if weighted_feat is not None:
                y_weighted_x_reference_features.append(weighted_feat)
        return {
            "y_last_remote_features": [weighted_mesg],
        }

    def flow_rescale(self, prev_flow, each_x_reference_features):
        if prev_flow is None:
            prev_flow = torch.zeros_like(
                each_x_reference_features[:, :2]).to(device)
        else:
            up_scale_factor = each_x_reference_features.shape[-1] / \
                prev_flow.shape[-1]
            if up_scale_factor != 1:
                prev_flow = F.interpolate(prev_flow, scale_factor=up_scale_factor, mode="bilinear",
                                          align_corners=False, recompute_scale_factor=False) * up_scale_factor
        return prev_flow
