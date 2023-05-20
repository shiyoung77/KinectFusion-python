# cd ..
# python -m kf_pycuda.reconstruct_with_poses

import os
import argparse
import json
from pprint import pprint
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from .tsdf_lib import TSDFVolume
from .kf_utils import create_pcd, vis_pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='~/t7/custom_room')
    parser.add_argument("-v", "--video", type=str, default="lab_scan2-Shiyang")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tsdf_voxel_size", type=float, default=0.015)
    parser.add_argument("--tsdf_trunc_margin", type=float, default=0.1)
    parser.add_argument("--output_voxel_size", type=float, default=0.02)
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
    cfg['tsdf_voxel_size'] = args.tsdf_voxel_size
    cfg['tsdf_trunc_margin'] = args.tsdf_trunc_margin
    pprint(cfg)

    combined_pts = []
    sampled_color_files = [color_files[i] for i in range(0, len(color_files), 100)]
    for color_file in tqdm(sampled_color_files):
        prefix = color_file.split('-')[0]
        depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        depth_im[depth_im > args.depth_trunc] = 0
        cam_pose_path = str(video_folder / 'poses' / f'{prefix}-pose.txt')
        cam_pose = np.loadtxt(cam_pose_path)
        pcd = create_pcd(depth_im, cfg['cam_intr'])
        pcd.transform(cam_pose)
        combined_pts.append(np.asarray(pcd.points))
    combined_pts = np.concatenate(combined_pts, axis=0)
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_pts)

    print("Computing minimal bounding box...")
    bbox = combined_pcd.get_minimal_oriented_bounding_box(robust=True)
    bbox.color = (1, 0, 0)
    vis_pcd([combined_pcd, bbox])

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = bbox.R
    extrinsic[:3, 3] = bbox.center
    extrinsic = np.linalg.inv(extrinsic)
    combined_pcd.transform(extrinsic)

    dx, dy, dz = bbox.extent
    volume_bounds = np.array([
        [-dx, dx],
        [-dy, dy],
        [-dz, dz]
    ])

    tsdf_volume = TSDFVolume(vol_bnds=volume_bounds,
                             voxel_size=cfg['tsdf_voxel_size'],
                             trunc_margin=cfg['tsdf_trunc_margin'])

    # Update TSDF volume
    sampled_color_files = [color_files[i] for i in range(0, len(color_files), args.stride)]
    for color_file in tqdm(sampled_color_files):
        color_im_path = str(video_folder / 'color' / f'{color_file}')
        prefix = color_file.split('-')[0]
        depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        try:
            depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        except AttributeError:
            print(f"Failed to read {depth_im_path}")
            continue
        depth_im[depth_im > args.depth_trunc] = 0

        cam_pose_path = str(video_folder / 'poses' / f'{prefix}-pose.txt')
        cam_pose = np.loadtxt(cam_pose_path)
        cam_pose = extrinsic @ cam_pose
        tsdf_volume.integrate(color_im, depth_im, cfg['cam_intr'], cam_pose)

    scan = tsdf_volume.get_surface_cloud_marching_cubes()
    scan.transform(np.linalg.inv(extrinsic))
    vis_pcd(scan)

    output_path = video_folder / args.output_folder / f'scan-{args.output_voxel_size:.3f}.pcd'
    scan_down = scan.voxel_down_sample(args.output_voxel_size)
    o3d.io.write_point_cloud(str(output_path), scan_down)
    print(f"Output point cloud voxel size {args.output_voxel_size:.3f}m")
    print(f"Saved point cloud to {output_path}")


if __name__ == '__main__':
    main()
