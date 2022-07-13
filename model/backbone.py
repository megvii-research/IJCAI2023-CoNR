import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

import torch.nn as nn
import torch
from torchvision import models


class AdaptiveConcatPool2d(nn.Module):
    """
    Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    Source: Fastai. This code was taken from the fastai library at url
    https://github.com/fastai/fastai/blob/master/fastai/layers.py#L176
    """

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class MyNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(
            num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.norm(x)
        return x


def resnet_fastai(model, pretrained,  url, replace_first_layer=None, replace_maxpool_layer=None, progress=True, map_location=None, **kwargs):
    cut = -2
    s = model(pretrained=False, **kwargs)
    if replace_maxpool_layer is not None:
        s.maxpool = replace_maxpool_layer
    if replace_first_layer is not None:
        body = nn.Sequential(replace_first_layer, *list(s.children())[1:cut])
    else:
        body = nn.Sequential(*list(s.children())[:cut])

    if pretrained:
        state = torch.hub.load_state_dict_from_url(url,
                                                   progress=progress, map_location=map_location)
        if replace_first_layer is not None:
            for each in list(state.keys()).copy():
                if each.find("0.0.") == 0:
                    del state[each]
        body_tail = nn.Sequential(body)
        ret = body_tail.load_state_dict(state, strict=False)
    return body


def get_backbone(name, pretrained=True, map_location=None):
    """ Loading backbone, defining names for skip-connections and encoder output. """

    first_layer_for_4chn = nn.Conv2d(
        4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    max_pool_layer_replace = nn.Conv2d(
        64, 64, kernel_size=3, stride=2, padding=1, bias=False)
    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    if name == 'resnet18-4':
        backbone = models.resnet18(pretrained=pretrained)
        backbone.conv1 = first_layer_for_4chn
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=False, norm_layer=MyNorm)
        backbone.maxpool = max_pool_layer_replace
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
    elif name == 'resnet18_danbo-4':
        backbone = resnet_fastai(models.resnet18, url="https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet18-3f77756f.pth",
                                 pretrained=pretrained, map_location=map_location, norm_layer=MyNorm, replace_first_layer=first_layer_for_4chn)
    elif name == 'resnet50_danbo':
        backbone = resnet_fastai(models.resnet50, url="https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
                                 pretrained=pretrained, map_location=map_location, norm_layer=MyNorm, replace_maxpool_layer=max_pool_layer_replace)
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    else:
        raise NotImplemented(
            '{} backbone model is not implemented so far.'.format(name))
    #print(backbone)
    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1',
                         'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented(
            '{} backbone model is not implemented so far.'.format(name))
    if name.find('_danbo') > 0:
        feature_names = [None, '2', '4', '5', '6']
        backbone_output = '7'
    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = MyNorm(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = MyNorm(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = MyNorm(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class ResEncUnet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name,
                 pretrained=True,
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(512, 256, 128, 64, 32),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_instancenorm=True,
                 map_location=None
                 ):
        super(ResEncUnet, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(
            backbone_name, pretrained=pretrained, map_location=map_location)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        # avoiding having more blocks than skip connections
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_instancenorm))
        self.final_conv = nn.Conv2d(
            decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

    def freeze_encoder(self):
        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input, ret_parser_out=True):
        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*input)
        output_feature = [x]
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            if skip_features is not None:
                output_feature.append(skip_features)
            if ret_parser_out:
                x = upsample_block(x, skip_features)
        if ret_parser_out:
            x = self.final_conv(x)
            # apply sigmoid later
        else:
            x = None

        return x, output_feature

    def forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):
        """ Getting the number of channels at skip connections and at the output of the encoder. """
        if self.backbone_name.find("-4") > 0:
            x = torch.zeros(1, 4, 224, 224)
        else:
            x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith(
            'vgg') or self.backbone_name == 'unet_encoder'
        # only VGG has features at full resolution
        channels = [] if has_fullres_features else [0]

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels
