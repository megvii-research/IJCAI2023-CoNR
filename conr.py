
import os
import torch

from model.backbone import ResEncUnet

from model.shader import CINN
from model.decoder_small import RGBADecoderNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def UDPClip(x):
    return torch.clamp(x, min=0, max=1)  # NCHW


class CoNR():
    def __init__(self, args):
        self.args = args

        self.udpparsernet = ResEncUnet(
            backbone_name='resnet50_danbo',
            classes=4,
            pretrained=(args.local_rank == 0),
            parametric_upsampling=True,
            decoder_filters=(512, 384, 256, 128, 32),
            map_location=device
        )
        self.target_pose_encoder = ResEncUnet(
            backbone_name='resnet18_danbo-4',
            classes=1,
            pretrained=(args.local_rank == 0),
            parametric_upsampling=True,
            decoder_filters=(512, 384, 256, 128, 32),
            map_location=device
        )
        self.DIM_SHADER_REFERENCE = 4
        self.shader = CINN(self.DIM_SHADER_REFERENCE)
        self.rgbadecodernet = RGBADecoderNet(
        )
        self.device()
        self.parser_ckpt = None

    def dist(self):
        args = self.args
        if args.distributed:
            self.udpparsernet = torch.nn.parallel.DistributedDataParallel(
                self.udpparsernet,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )
            self.target_pose_encoder = torch.nn.parallel.DistributedDataParallel(
                self.target_pose_encoder,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )
            self.shader = torch.nn.parallel.DistributedDataParallel(
                self.shader,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=True
            )

            self.rgbadecodernet = torch.nn.parallel.DistributedDataParallel(
                self.rgbadecodernet,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=True
            )

    def load_model(self, path):
        self.udpparsernet.load_state_dict(
            torch.load('{}/udpparsernet.pth'.format(path), map_location=device))
        self.target_pose_encoder.load_state_dict(
            torch.load('{}/target_pose_encoder.pth'.format(path), map_location=device))
        self.shader.load_state_dict(
            torch.load('{}/shader.pth'.format(path), map_location=device))
        self.rgbadecodernet.load_state_dict(
            torch.load('{}/rgbadecodernet.pth'.format(path), map_location=device))

    def save_model(self, ite_num):
        self._save_pth(self.udpparsernet,
                       model_name="udpparsernet", ite_num=ite_num)
        self._save_pth(self.target_pose_encoder,
                       model_name="target_pose_encoder", ite_num=ite_num)
        self._save_pth(self.shader,
                       model_name="shader", ite_num=ite_num)
        self._save_pth(self.rgbadecodernet,
                       model_name="rgbadecodernet", ite_num=ite_num)

    def _save_pth(self, net, model_name, ite_num):
        args = self.args
        to_save = None
        if args.distributed:
            if args.local_rank == 0:
                to_save = net.module.state_dict()
        else:
            to_save = net.state_dict()
        if to_save:
            model_dir = os.path.join(
                os.getcwd(), 'saved_models', args.model_name + os.sep + "checkpoints" + os.sep + "itr_%d" % (ite_num)+os.sep)

            os.makedirs(model_dir, exist_ok=True)
            torch.save(to_save, model_dir + model_name + ".pth")

    def train(self):
        self.udpparsernet.train()
        self.target_pose_encoder.train()
        self.shader.train()
        self.rgbadecodernet.train()

    def eval(self):
        self.udpparsernet.eval()
        self.target_pose_encoder.eval()
        self.shader.eval()
        self.rgbadecodernet.eval()

    def device(self):
        self.udpparsernet.to(device)
        self.target_pose_encoder.to(device)
        self.shader.to(device)
        self.rgbadecodernet.to(device)

    def data_norm_image(self, data):

        with torch.cuda.amp.autocast(enabled=False):
            for name in ["character_labels", "pose_label"]:
                if name in data:
                    data[name] = data[name].to(
                        device, non_blocking=True).float()
            for name in ["pose_images", "pose_mask", "character_images", "character_masks"]:
                if name in data:
                    data[name] = data[name].to(
                        device, non_blocking=True).float() / 255.0
            if "pose_images" in data:
                data["num_pose_images"] = data["pose_images"].shape[1]
                data["num_samples"] = data["pose_images"].shape[0]
            if "character_images" in data:
                data["num_character_images"] = data["character_images"].shape[1]
                data["num_samples"] = data["character_images"].shape[0]
            if "pose_images" in data and "character_images" in data:
                assert (data["pose_images"].shape[0] ==
                        data["character_images"].shape[0])
        return data

    def reset_charactersheet(self):
        self.parser_ckpt = None

    def model_step(self, data, training=False):
        self.eval()
        with torch.cuda.amp.autocast(enabled=False):
            pred = {}
            if self.parser_ckpt:
                pred["parser"] = self.parser_ckpt
            else:
                pred = self.character_parser_forward(data, pred)
                self.parser_ckpt = pred["parser"]
            pred = self.pose_parser_sc_forward(data, pred)
            pred = self.shader_pose_encoder_forward(data, pred)
            pred = self.shader_forward(data, pred)
        return pred

    def shader_forward(self, data, pred={}):
        assert ("num_character_images" in data), "ERROR: No Character Sheet input."

        character_images_rgb_nmchw, num_character_images = data[
            "character_images"], data["num_character_images"]
        # build  x_reference_rgb_a_sudp in the draw call
        shader_character_a_nmchw = data["character_masks"]
        assert torch.any(torch.mean(shader_character_a_nmchw, (0, 2, 3, 4)) >= 0.95) == False, "ERROR: \
                No transparent area found in the image, PLEASE separate the foreground of input character sheets.\
                The website waifucutout.com is recommended to automatically cut out the foreground."
        
        if shader_character_a_nmchw is None:
            shader_character_a_nmchw = pred["parser"]["pred"][:, :, 3:4, :, :]
        x_reference_rgb_a = torch.cat([shader_character_a_nmchw[:, :, :, :, :] * character_images_rgb_nmchw[:, :, :, :, :],
                                       shader_character_a_nmchw[:,
                                                                :, :, :, :],

                                       ], 2)
        assert (x_reference_rgb_a.shape[2] == self.DIM_SHADER_REFERENCE)
        # build  x_reference_features in the draw call
        x_reference_features = pred["parser"]["features"]
        # run cinn shader
        retdic = self.shader(
            pred["shader"]["target_pose_features"], x_reference_rgb_a, x_reference_features)
        pred["shader"].update(retdic)

        # decode rgba
        if True:
            dec_out = self.rgbadecodernet(
                retdic["y_last_remote_features"])
            y_weighted_x_reference_RGB = dec_out[:, 0:3, :, :]
            y_weighted_mask_A = dec_out[:, 3:4, :, :]
        y_weighted_warp_decoded_rgba = torch.cat(
            (y_weighted_x_reference_RGB*y_weighted_mask_A, y_weighted_mask_A), dim=1
        )
        assert(y_weighted_warp_decoded_rgba.shape[1] == 4)
        assert(
            y_weighted_warp_decoded_rgba.shape[-1] == character_images_rgb_nmchw.shape[-1])
        # apply decoded mask to decoded rgb, finishing the draw call
        pred["shader"]["y_weighted_warp_decoded_rgba"] = y_weighted_warp_decoded_rgba
        return pred

    def character_parser_forward(self, data, pred={}):
        if not("num_character_images" in data and "character_images" in data):
            return pred
        pred["parser"] = {"pred": None}  # create output

        inputs_rgb_nmchw, num_samples, num_character_images = data[
            "character_images"],  data["num_samples"], data["num_character_images"]
        inputs_rgb_fchw = inputs_rgb_nmchw.view(
            (num_samples * num_character_images, inputs_rgb_nmchw.shape[2], inputs_rgb_nmchw.shape[3], inputs_rgb_nmchw.shape[4]))

        encoder_out, features = self.udpparsernet(
            (inputs_rgb_fchw-0.6)/0.2970)

        pred["parser"]["features"] = [features_out.view(
            (num_samples, num_character_images, features_out.shape[1], features_out.shape[2], features_out.shape[3])) for features_out in features]

        if (encoder_out is not None):

            pred["parser"]["pred"] = UDPClip(encoder_out.view(
                (num_samples, num_character_images, encoder_out.shape[1], encoder_out.shape[2], encoder_out.shape[3])))

        return pred

    def pose_parser_sc_forward(self, data, pred={}):
        if not("num_pose_images" in data and "pose_images" in data):
            return pred
        inputs_aug_rgb_nmchw, num_samples, num_pose_images = data[
            "pose_images"],  data["num_samples"], data["num_pose_images"]
        inputs_aug_rgb_fchw = inputs_aug_rgb_nmchw.view(
            (num_samples * num_pose_images, inputs_aug_rgb_nmchw.shape[2], inputs_aug_rgb_nmchw.shape[3], inputs_aug_rgb_nmchw.shape[4]))

        encoder_out, _ = self.udpparsernet(
            (inputs_aug_rgb_fchw-0.6)/0.2970)

        encoder_out = encoder_out.view(
            (num_samples, num_pose_images, encoder_out.shape[1], encoder_out.shape[2], encoder_out.shape[3]))

        # apply sigmoid after eval loss
        pred["pose_parser"] = {"pred":UDPClip(encoder_out)}
        

        return pred

    def shader_pose_encoder_forward(self, data, pred={}):
        pred["shader"] = {}  # create output
        if "pose_images" in data:
            pose_images_rgb_nmchw = data["pose_images"]
            target_gt_rgb = pose_images_rgb_nmchw[:, 0, :, :, :]
            pred["shader"]["target_gt_rgb"] = target_gt_rgb

        shader_target_a = None
        if "pose_mask" in data:
            pred["shader"]["target_gt_a"] = data["pose_mask"]
            shader_target_a = data["pose_mask"]

        shader_target_sudp = None
        if "pose_label" in data:
            shader_target_sudp = data["pose_label"][:, :3, :, :]

        if self.args.test_pose_use_parser_udp:
            shader_target_sudp = None
        if shader_target_sudp is None:
            shader_target_sudp = pred["pose_parser"]["pred"][:, 0:3, :, :]

        if shader_target_a is None:
            shader_target_a = pred["pose_parser"]["pred"][:, 3:4, :, :]

        # build x_target_sudp_a in the draw call
        x_target_sudp_a = torch.cat((
            shader_target_sudp*shader_target_a,
            shader_target_a
        ), 1)
        pred["shader"].update({
            "x_target_sudp_a": x_target_sudp_a
        })
        _, features = self.target_pose_encoder(
            (x_target_sudp_a-0.6)/0.2970, ret_parser_out=False)

        pred["shader"]["target_pose_features"] = features
        return pred