import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
cv2.setNumThreads(1)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class RandomResizedCropWithAutoCenteringAndZeroPadding (object):
    def __init__(self, output_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), center_jitter=(0.1, 0.1), size_from_alpha_mask=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        assert isinstance(scale,  tuple)
        assert isinstance(ratio,  tuple)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")
        self.size_from_alpha_mask = size_from_alpha_mask
        self.scale = scale
        self.ratio = ratio
        assert isinstance(center_jitter,  tuple)
        self.center_jitter = center_jitter

    def __call__(self, sample):
        imidx, image = sample['imidx'], sample["image_np"]
        if "labels" in sample:
            label = sample["labels"]
        else:
            label = None

        im_h, im_w = image.shape[:2]
        if self.size_from_alpha_mask and image.shape[2] == 4:
            # compute bbox from alpha mask
            bbox_left, bbox_top, bbox_w, bbox_h = cv2.boundingRect(
                (image[:, :, 3] > 0).astype(np.uint8))
        else:
            bbox_left, bbox_top = 0, 0
            bbox_h, bbox_w = image.shape[:2]
        if bbox_h <= 1 and bbox_w <= 1:
            sample["bad"] = 0
        else:
            # detect too small image here
            alpha_varea = np.sum((image[:, :, 3] > 0).astype(np.uint8))
            image_area = image.shape[0]*image.shape[1]
            if alpha_varea/image_area < 0.001:
                sample["bad"] = alpha_varea
                # detect bad image
        if "bad" in sample:
            # baddata_dir = os.path.join(os.getcwd(), 'test_data', "baddata" + os.sep)
            # save_output(str(imidx)+".png",image,label,baddata_dir)
            bbox_h, bbox_w = image.shape[:2]
            sample["image_np"] = np.zeros(
                [self.output_size[0], self.output_size[1], image.shape[2]], dtype=image.dtype)
            if label is not None:
                sample["labels"] = np.zeros(
                    [self.output_size[0], self.output_size[1], 4], dtype=label.dtype)

            return sample

        # compute default area by making sure output_size contains bbox_w * bbox_h

        jitter_h = np.random.uniform(-bbox_h *
                                     self.center_jitter[0], bbox_h*self.center_jitter[0])
        jitter_w = np.random.uniform(-bbox_w *
                                     self.center_jitter[1], bbox_w*self.center_jitter[1])

        # h/w
        target_aspect_ratio = np.exp(
            np.log(self.output_size[0]/self.output_size[1]) +
            np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
        )

        source_aspect_ratio = bbox_h/bbox_w

        if target_aspect_ratio < source_aspect_ratio:
            # same w, target has larger h, use h to align
            target_height = bbox_h * \
                np.random.uniform(self.scale[0], self.scale[1])
            virtual_h = int(
                round(target_height))
            virtual_w = int(
                round(target_height / target_aspect_ratio))  # h/w
        else:
            # same w, source has larger h, use w to align
            target_width = bbox_w * \
                np.random.uniform(self.scale[0], self.scale[1])
            virtual_h = int(
                round(target_width * target_aspect_ratio))  # h/w
            virtual_w = int(
                round(target_width))

        # print("required aspect ratio:",      target_aspect_ratio)

        virtual_top = int(round(bbox_top + jitter_h - (virtual_h-bbox_h)/2))
        virutal_left = int(round(bbox_left + jitter_w - (virtual_w-bbox_w)/2))

        if virtual_top < 0:
            top_padding = abs(virtual_top)
            crop_top = 0
        else:
            top_padding = 0
            crop_top = virtual_top
        if virutal_left < 0:
            left_padding = abs(virutal_left)
            crop_left = 0
        else:
            left_padding = 0
            crop_left = virutal_left
        if virtual_top+virtual_h > im_h:
            bottom_padding = abs(im_h-(virtual_top+virtual_h))
            crop_bottom = im_h
        else:
            bottom_padding = 0
            crop_bottom = virtual_top+virtual_h
        if virutal_left+virtual_w > im_w:
            right_padding = abs(im_w-(virutal_left+virtual_w))
            crop_right = im_w
        else:
            right_padding = 0
            crop_right = virutal_left+virtual_w
        # crop

        image = image[crop_top:crop_bottom, crop_left:  crop_right]
        if label is not None:
            label = label[crop_top:crop_bottom, crop_left:  crop_right]

        # pad
        if top_padding + bottom_padding + left_padding + right_padding > 0:
            padding = ((top_padding, bottom_padding),
                       (left_padding, right_padding), (0, 0))
            # print("padding", padding)
            image = np.pad(image, padding, mode='constant')
            if label is not None:
                label = np.pad(label, padding, mode='constant')

        if image.shape[0]/image.shape[1] - virtual_h/virtual_w > 0.001:
            print("virtual aspect ratio:",  virtual_h/virtual_w)
            print("image aspect ratio:", image.shape[0]/image.shape[1])
        assert (image.shape[0]/image.shape[1] - virtual_h/virtual_w < 0.001)
        sample["crop"] = np.array(
            [im_h, im_w,  crop_top, crop_bottom, crop_left, crop_right, top_padding, bottom_padding, left_padding, right_padding, image.shape[0], image.shape[1]])

        # resize
        if self.output_size[1] != image.shape[1] or self.output_size[0] != image.shape[0]:
            if self.output_size[1] > image.shape[1] and self.output_size[0] > image.shape[0]:
                # enlarging
                image = cv2.resize(
                    image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            else:
                # shrinking
                image = cv2.resize(
                    image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_AREA)

            if label is not None:
                label = cv2.resize(label, (self.output_size[1], self.output_size[0]),
                                   interpolation=cv2.INTER_NEAREST_EXACT)

        assert image.shape[0] == self.output_size[0] and image.shape[1] == self.output_size[1]
        sample['imidx'], sample["image_np"] = imidx, image
        if label is not None:
            assert label.shape[0] == self.output_size[0] and label.shape[1] == self.output_size[1]
            sample["labels"] = label

        return sample


class FileDataset(Dataset):
    def __init__(self, image_names_list, fg_img_lbl_transform=None, shader_pose_use_gt_udp_test=True, shader_target_use_gt_rgb_debug=False):
        self.image_name_list = image_names_list
        self.fg_img_lbl_transform = fg_img_lbl_transform
        self.shader_pose_use_gt_udp_test = shader_pose_use_gt_udp_test
        self.shader_target_use_gt_rgb_debug = shader_target_use_gt_rgb_debug

    def __len__(self):
        return len(self.image_name_list)

    def get_gt_from_disk(self, idx, imname, read_label):
        if read_label:
            # read label
            with open(imname, mode="rb") as bio:
                if imname.find(".npz") > 0:
                    label_np = np.load(bio, allow_pickle=True)[
                        'i'].astype(np.float32, copy=False)
                else:
                    label_np = cv2.cvtColor(cv2.imdecode(np.frombuffer(bio.read(
                    ), np.uint8),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            assert (4 == label_np.shape[2])
            # fake image out of valid label
            image_np = (label_np*255).clip(0, 255).astype(np.uint8, copy=False)
            # assemble sample
            sample = {'imidx': np.array(
                [idx]),   "image_np": image_np, "labels": label_np}

        else:
            # read image as unit8
            with open(imname, mode="rb") as bio:
                image_np = cv2.cvtColor(cv2.imdecode(np.frombuffer(
                    bio.read(), np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                # image_np = Image.open(bio)
                # image_np = np.array(image_np)
            assert (3 == len(image_np.shape))
            if (image_np.shape[2] == 4):
                mask_np = image_np[:, :, 3:4]
                image_np = (image_np[:, :, :3] *
                            (image_np[:, :, 3][:, :, np.newaxis]/255.0)).clip(0, 255).astype(np.uint8, copy=False)
            elif (image_np.shape[2] == 3):
                # generate a fake mask
                # Fool-proofing
                mask_np = np.ones(
                    (image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8)*255
                print("WARN: transparent background is preferred for image ", imname)
            else:
                raise ValueError("weird shape of image ", imname, image_np)
            image_np = np.concatenate((image_np, mask_np), axis=2)
            sample = {'imidx': np.array(
                [idx]),   "image_np": image_np}

        # apply fg_img_lbl_transform
        if self.fg_img_lbl_transform:
            sample = self.fg_img_lbl_transform(sample)

        if "labels" in sample:
            # return UDP as 4chn XYZV float tensor
            sample["labels"] = torch.from_numpy(
                sample["labels"].transpose((2, 0, 1)))
            assert (sample["labels"].dtype == torch.float32)

        if "image_np" in sample:
            # return image as 3chn RGB uint8 tensor and 1chn A uint8 tensor
            sample["mask"] = torch.from_numpy(
                sample["image_np"][:, :, 3:4].transpose((2, 0, 1)))
            assert (sample["mask"].dtype == torch.uint8)
            sample["image"] = torch.from_numpy(
                sample["image_np"][:, :, :3].transpose((2, 0, 1)))

            assert (sample["image"].dtype == torch.uint8)
            del sample["image_np"]
        return sample

    def __getitem__(self, idx):
        sample = {
            'imidx': np.array([idx])}
        target = self.get_gt_from_disk(
            idx, imname=self.image_name_list[idx][0], read_label=self.shader_pose_use_gt_udp_test)
        if self.shader_target_use_gt_rgb_debug:
            sample["pose_images"] = torch.stack([target["image"]])
            sample["pose_mask"] = target["mask"]
        elif self.shader_pose_use_gt_udp_test:
            sample["pose_label"] = target["labels"]
            sample["pose_mask"] = target["mask"]
        else:
            sample["pose_images"] = torch.stack([target["image"]])
        if "crop" in target:
            sample["pose_crop"] = target["crop"]
        character_images = []
        character_masks = []
        for i in range(1, len(self.image_name_list[idx])):
            source = self.get_gt_from_disk(
                idx, self.image_name_list[idx][i], read_label=False)
            character_images.append(source["image"])
            character_masks.append(source["mask"])
        character_images = torch.stack(character_images)
        character_masks = torch.stack(character_masks)
        sample.update({
            "character_images": character_images,
            "character_masks": character_masks
        })
        # do not make fake labels in inference
        return sample