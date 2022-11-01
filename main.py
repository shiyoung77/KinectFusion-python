# cd ..
# python -m kf_pycuda.main

import os
import argparse
import numpy as np
import cv2
import json
from tqdm.contrib import tenumerate

from .kinect_fusion import KinectFusion
from .kf_config import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos')
    parser.add_argument("-v", "--video", type=str, default="kitchen_0001")
    parser.add_argument("--color_im_ext", type=str, default="jpg")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=2.0)
    parser.add_argument("--vis", default=False, action="store_true")
    args = parser.parse_args()

    video_folder = os.path.join(args.dataset, args.video)
    color_folder = os.path.join(video_folder, 'color')
    prefix_list = sorted([i.split('-')[0] for i in os.listdir(color_folder)])

    data_cfg_path = os.path.join(video_folder, 'config.json')
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['cam_intr'] = np.asarray(cfg['cam_intr'])

    kf_cfg = get_config()
    cfg.update(kf_cfg)

    kf = KinectFusion(cfg=cfg)

    # initialize TSDF with the first frame
    color_im_path = os.path.join(video_folder, 'color', prefix_list[args.start_frame] + f'-color.{args.color_im_ext}')
    depth_im_path = os.path.join(video_folder, 'depth', prefix_list[args.start_frame] + '-depth.png')
    color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
    depth_im[depth_im > args.depth_trunc] = 0
    kf.initialize_tsdf_volume(color_im, depth_im, visualize=args.vis)

    # Update TSDF volume
    for _, prefix in tenumerate(prefix_list[1:]):
        color_im_path = os.path.join(video_folder, 'color', prefix + f'-color.{args.color_im_ext}')
        depth_im_path = os.path.join(video_folder, 'depth', prefix + '-depth.png')
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        depth_im[depth_im > args.depth_trunc] = 0
        kf.update(color_im, depth_im)

    # output_dir = os.path.join(video_folder, 'recon')
    # kf.save(output_dir, voxel_size=0.001)
