import os
import json
import argparse
from time import perf_counter
from typing import Dict, List

import cv2
import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import cdist, pdist
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm

import kf_utils


def read_frames(filepath: os.PathLike) -> Dict[str, List]:
    with open(filepath, 'r') as fp:
        frame_to_pose = json.load(fp)
    return frame_to_pose


def visualize_poses(frame_to_pose: Dict) -> None:
    meshes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    cam_positions = []
    for frame, cam_pose_inv in sorted(frame_to_pose.items()):
        cam_pose = la.inv(np.array(cam_pose_inv))
        cam_positions.append(cam_pose[:3, 3])
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame_mesh.transform(cam_pose)
        meshes.append(frame_mesh)

    cam_positions = np.stack(cam_positions)
    tic = perf_counter()
    distances = cdist(cam_positions, cam_positions)
    print(f"cdist takes {perf_counter() - tic}s")
    print(distances.shape)
    print(distances.max())
    print(distances.min())

    # ind = np.unravel_index(np.argmax(distances), shape=distances.shape)

    # tic = perf_counter()
    # distances = pdist(cam_positions)
    # print(f"pdist takes {perf_counter() - tic}s")
    # print(distances.shape)
    # print(distances.max())
    # print(distances.min())
    o3d.visualization.draw_geometries(meshes)


def get_frustum(video_folder, frame_to_pose: Dict):
    data_cfg_path = os.path.join(video_folder, 'config.json')
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cam_intr = np.asarray(cfg['cam_intr'])
    depth_scale = cfg['depth_scale']
    meshes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)]
    # pcds = []
    pcds = o3d.geometry.PointCloud()
    for idx, (frame, cam_pose_inv) in enumerate(tqdm(frame_to_pose.items(), disable=True)):
        cam_pose = la.inv(np.array(cam_pose_inv))
        prefix = int(frame.split('.')[0])
        color_im_path = os.path.join(video_folder, 'color', f"{prefix:04d}-color.jpg")
        depth_im_path = os.path.join(video_folder, 'depth', f'{prefix:04d}-depth.png')

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        meshes.append(world_frame.transform(cam_pose))

        # if idx % 10 == 0:
        if idx < 500:
            color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
            depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
            # depth_im[depth_im > args.depth_trunc] = 0
            pcd = kf_utils.create_pcd(depth_im, cam_intr, color_im, depth_scale=0.5, depth_trunc=10, cam_extr=la.inv(cam_pose))
            pcds += pcd
            # pcds.append(pcd)
            if idx % 50 == 0:
                pcds = pcds.voxel_down_sample(0.05)

    # o3d.visualization.draw_geometries(meshes)
    o3d.visualization.draw_geometries(meshes + [pcds])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos/')
    parser.add_argument("-v", "--video", type=str, default="kitchen_0001")
    parser.add_argument("--depth_trunc", type=float, default=2.0)
    parser.add_argument("--vis", default=False, action="store_true")
    args = parser.parse_args()

    video_folder = os.path.join(args.dataset, args.video)

    pose_file = "/home/lsy/dataset/nerf/collected_videos/kitchen_0001/colmap_text/cam_poses.json"
    frames = read_frames(pose_file)
    print(f"{len(frames) = }")

    # visualize_poses(frames)
    get_frustum(video_folder, frames)