# cd ..
# python -m kf_pycuda.main

import os
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
import open3d as o3d

from .kinect_fusion import KinectFusion
from .kf_config import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos')
    parser.add_argument("-v", "--video", type=str, default="kitchen_0002")
    parser.add_argument("--color_im_ext", type=str, default="jpg")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    video_folder = os.path.join(args.dataset, args.video)
    color_folder = os.path.join(video_folder, 'color')
    frame_ids = sorted([int(i.split('-')[0]) for i in os.listdir(color_folder)])

    if args.end_frame == -1:
        end_frame = frame_ids[-1]
    else:
        end_frame = min(frame_ids[-1], args.end_frame)

    data_cfg_path = os.path.join(video_folder, 'config.json')
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['cam_intr'] = np.asarray(cfg['cam_intr'])

    kf_cfg = get_config()
    cfg.update(kf_cfg)

    kf = KinectFusion(cfg=cfg)

    # initialize TSDF with the first frame
    color_im_path = os.path.join(video_folder, 'color', f'{args.start_frame:04d}-color.{args.color_im_ext}')
    depth_im_path = os.path.join(video_folder, 'depth', f'{args.start_frame:04d}-depth.png')
    color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
    depth_im[depth_im > 2] = 0
    kf.initialize_tsdf_volume(color_im, depth_im, visualize=True)

    # Update TSDF volume
    for frame_id in tqdm(range(args.start_frame + 1, end_frame + 1, args.stride)):
        color_im_path = os.path.join(video_folder, 'color', f'{frame_id:04d}-color.{args.color_im_ext}')
        depth_im_path = os.path.join(video_folder, 'depth', f'{frame_id:04d}-depth.png')
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        depth_im[depth_im > args.depth_trunc] = 0
        kf.update(color_im, depth_im)
    
    cam_poses = np.stack(kf.cam_poses)
    cam_frames = []
    for pose in cam_poses:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(pose)
        cam_frames.append(cam_frame)

    recon = kf.tsdf_volume.get_surface_cloud_marching_cubes(voxel_size=0.005)
    o3d.visualization.draw_geometries([kf.vol_box, recon] + cam_frames)

    if args.save:
        output_dir = os.path.join(video_folder, 'recon')
        recon_pcd = kf.save(output_dir, voxel_size=0.005)
