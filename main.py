# cd ..
# python -m kf_pycuda.main

import os
import argparse
import json
from pprint import pprint
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from .kinect_fusion import KinectFusion
from .kf_config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos')
    parser.add_argument("-v", "--video", type=str, default="recording")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output_folder", type=str, default=".")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    video_folder = dataset / args.video
    color_folder = video_folder / 'color'
    if not os.path.isdir(color_folder):
        print(f"{color_folder} doesn't exist.")
        exit(1)

    color_files = sorted(os.listdir(color_folder))
    if args.end_frame == -1:
        args.end_frame = len(color_files)
    color_files = color_files[args.start_frame:args.end_frame]

    data_cfg_path = video_folder / 'config.json'
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['cam_intr'] = np.asarray(cfg['cam_intr'])

    kf_cfg = get_config()
    cfg.update(kf_cfg)
    pprint(cfg)

    kf = KinectFusion(cfg=cfg)

    # initialize TSDF with the first frame
    color_im_path = str(video_folder / 'color' / f'{color_files[0]}')
    prefix = color_files[0].split('-')[0]
    depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
    color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
    depth_im[depth_im > 2] = 0
    kf.initialize_tsdf_volume(color_im, depth_im, visualize=True)

    # Update TSDF volume
    for color_file in tqdm(color_files[1:]):
        color_im_path = str(video_folder / 'color' / f'{color_file}')
        prefix = color_file.split('-')[0]
        depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        depth_im[depth_im > args.depth_trunc] = 0
        kf.update(color_im, depth_im)

    cam_frames = []
    for cam_pose in kf.cam_poses:
        cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        cam_frame.transform(cam_pose)
        cam_frames.append(cam_frame)
    recon = kf.tsdf_volume.get_surface_cloud_marching_cubes()
    o3d.visualization.draw_geometries([kf.vol_box, recon] + cam_frames)

    if args.output_folder:
        output_dir = video_folder / args.output_folder
        kf.save(output_dir)

        # save camera poses in txt files.
        assert len(kf.cam_poses) == len(color_files)
        output_pose_dir = output_dir / 'poses'
        output_pose_dir.mkdir(parents=True, exist_ok=True)
        for color_file, cam_pose in zip(color_files, kf.cam_poses):
            prefix = color_file.split('-')[0]
            pose_file = output_pose_dir / f'{prefix}-pose.txt'
            np.savetxt(pose_file, cam_pose, fmt='%.6f')


if __name__ == '__main__':
    main()
