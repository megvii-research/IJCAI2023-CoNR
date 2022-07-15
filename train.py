import argparse
import os
import time
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import (FileDataset,
                         RandomResizedCropWithAutoCenteringAndZeroPadding)
from torch.utils.data.distributed import DistributedSampler
from conr import CoNR
from tqdm import tqdm

def data_sampler(dataset, shuffle, distributed):

    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)

def save_output(image_name, inputs_v, d_dir=".", crop=None):
    import cv2

    inputs_v = inputs_v.detach().squeeze()
    input_np = torch.clamp(inputs_v*255, 0, 255).byte().cpu().numpy().transpose(
        (1, 2, 0))
    # cv2.setNumThreads(1)
    out_render_scale = cv2.cvtColor(input_np, cv2.COLOR_RGBA2BGRA)
    if crop is not None:
        crop = crop.cpu().numpy()[0]
        output_img = np.zeros((crop[0], crop[1], 4), dtype=np.uint8)
        before_resize_scale = cv2.resize(
            out_render_scale, (crop[5]-crop[4]+crop[8]+crop[9], crop[3]-crop[2]+crop[6]+crop[7]), interpolation=cv2.INTER_AREA)  # w,h
        output_img[crop[2]:crop[3], crop[4]:crop[5]] = before_resize_scale[crop[6]:before_resize_scale.shape[0] -
                                                                           crop[7], crop[8]:before_resize_scale.shape[1]-crop[9]]
    else:
        output_img = out_render_scale
    cv2.imwrite(d_dir+"/"+image_name.split(os.sep)[-1]+'.png',
                output_img
                )


def test():
    source_names_list = []
    for name in sorted(os.listdir(args.test_input_person_images)):
        thissource = os.path.join(args.test_input_person_images, name)
        if os.path.isfile(thissource):
            source_names_list.append(thissource)
        if os.path.isdir(thissource):
            print("skipping empty folder :"+thissource)

    image_names_list = []
    for name in sorted(os.listdir(args.test_input_poses_images)):
        thistarget = os.path.join(args.test_input_poses_images, name)
        if os.path.isfile(thistarget):
            image_names_list.append([thistarget, *source_names_list])
        if os.path.isdir(thistarget):
            print("skipping folder :"+thistarget)
    print(image_names_list)

    print("---building models")
    conrmodel = CoNR(args)
    conrmodel.load_model(path=args.test_checkpoint_dir)
    conrmodel.dist()
    infer(args, conrmodel, image_names_list)


def infer(args, humanflowmodel, image_names_list):
    print("---test images: ", len(image_names_list))
    test_salobj_dataset = FileDataset(image_names_list=image_names_list,
                                      fg_img_lbl_transform=transforms.Compose([
                                          RandomResizedCropWithAutoCenteringAndZeroPadding(
                                              (args.dataloader_imgsize, args.dataloader_imgsize), scale=(1, 1), ratio=(1.0, 1.0), center_jitter=(0.0, 0.0)
                                          )]),
                                      shader_pose_use_gt_udp_test=not args.test_pose_use_parser_udp,
                                      shader_target_use_gt_rgb_debug=False
                                      )
    sampler = data_sampler(test_salobj_dataset, shuffle=False,
                           distributed=args.distributed)
    train_data = DataLoader(test_salobj_dataset,
                            batch_size=1,
                            shuffle=False,sampler=sampler, 
                            num_workers=args.dataloaders)

    # start testing

    train_num = train_data.__len__()
    time_stamp = time.time()
    prev_frame_rgb = []
    prev_frame_a = []
    
    pbar = tqdm(range(train_num), ncols=100)
    for i, data in enumerate(train_data):
        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        with torch.no_grad():
            data["character_images"] = torch.cat(
                [data["character_images"], *prev_frame_rgb], dim=1)
            data["character_masks"] = torch.cat(
                [data["character_masks"], *prev_frame_a], dim=1)
            data = humanflowmodel.data_norm_image(data)
            pred = humanflowmodel.model_step(data, training=False)
            # remember to call  humanflowmodel.reset_charactersheet() if you change character .

        train_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        if args.local_rank == 0:
            pbar.set_description(f"Epoch {i}/{train_num}")
            pbar.set_postfix({"data_time": data_time_interval, "train_time":train_time_interval})
            pbar.update(1)

        with torch.no_grad():

            if args.test_output_video:
                pred_img = pred["shader"]["y_weighted_warp_decoded_rgba"]
                save_output(
                    str(int(data["imidx"].cpu().item())), pred_img, args.test_output_dir, crop=data["pose_crop"])
            
            if args.test_output_udp:
                pred_img = pred["shader"]["x_target_sudp_a"]
                save_output(
                    "udp_"+str(int(data["imidx"].cpu().item())), pred_img, args.test_output_dir)


def build_args():
    parser = argparse.ArgumentParser()
    # distributed learning settings
    parser.add_argument("--world_size", type=int, default=1,
                        help='world size')
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local_rank, DON\'T change it')

    # model settings
    parser.add_argument('--dataloader_imgsize', type=int, default=256,
                        help='Input image size of the model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='minibatch size')
    parser.add_argument('--model_name', default='model_result',
                        help='Name of the experiment')
    parser.add_argument('--dataloaders', type=int, default=2,
                        help='Num of dataloaders')
    parser.add_argument('--mode', default="test", choices=['train', 'test'],
                        help='Training mode or Testing mode')

    # i/o settings
    parser.add_argument('--test_input_person_images',
                        type=str, default="./character_sheet/",
                        help='Directory to input character sheets')
    parser.add_argument('--test_input_poses_images', type=str,
                        default="./test_data/",
                        help='Directory to input UDP sequences or pose images')
    parser.add_argument('--test_checkpoint_dir', type=str,
                        default='./weights/',
                        help='Directory to model weights')
    parser.add_argument('--test_output_dir', type=str,
                        default="./results/",
                        help='Directory to output images')

    # output content settings
    parser.add_argument('--test_output_video', type=strtobool, default=True,
                        help='Whether to output the final result of CoNR, \
                              images will be output to test_output_dir while True.')
    parser.add_argument('--test_output_udp', type=strtobool, default=False,
                        help='Whether to output UDP generated from UDP detector, \
                              this is meaningful ONLY when test_input_poses_images \
                              is not UDP sequences but pose images. Meanwhile, \
                              test_pose_use_parser_udp need to be True')

    # UDP detector settings
    parser.add_argument('--test_pose_use_parser_udp',
                        type=strtobool, default=False, 
                        help='Whether to use UDP detector to generate UDP from pngs, \
                              pose input MUST be pose images instead of UDP sequences \
                              while True')

    args = parser.parse_args()

    args.distributed = (args.world_size > 1)
    if args.local_rank == 0:
        print("batch_size:", args.batch_size, flush=True)
    if args.distributed:
        if args.local_rank == 0:
            print("world_size: ", args.world_size)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        args.local_rank = 0
 
    return args


if __name__ == "__main__":
    args = build_args()
    test()