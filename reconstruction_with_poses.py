# cd ..
# python -m kf_pycuda.main

import os
import copy
import argparse
import numpy as np
import numpy.linalg as la
import cv2
import json
from tqdm import tqdm
import open3d as o3d

from .tsdf_lib import TSDFVolume
from . import kf_utils as utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/test/kitchen_0001/')
    parser.add_argument("-v", "--video", type=str, default="kitchen_0002")
    parser.add_argument("--color_im_ext", type=str, default="jpg")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.dataset, 'depth_files.txt'), 'r') as fp:
        depth_files = []
        line = fp.readline().strip()
        while line:
            depth_files.append(line)
            line = fp.readline().strip()

    with open('/home/lsy/dataset/collected_videos/kitchen_0001/config.json', 'r') as fp:
        data = json.load(fp)
    cam_intr = np.asarray(data['cam_intr'])

    poses = np.loadtxt(os.path.join(args.dataset, 'poses.txt'))
    coordinate_frames = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
    T_list = []
    positions = []
    for pose, depth_file in zip(poses, depth_files):
        pose = pose.reshape((3, 4))
        R = pose[:3, :3]
        R[:, 1:3] *= -1
        t = pose[:3, 3]
        positions.append(t)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T_list.append(T)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coord.transform(T)
        coordinate_frames.append(coord)

    positions = np.stack(positions)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    plane_frame, inlier_ratio = utils.timeit(utils.plane_detection_o3d)(pcd,
                                                                        max_iterations=500,
                                                                        inlier_thresh=0.005,
                                                                        visualize=False)

    extrinsic = la.inv(plane_frame)
    transformed_pcd = copy.deepcopy(pcd).transform(extrinsic)
    transformed_pts = np.array(transformed_pcd.points)
    vol_bnds = np.zeros((3, 2), dtype=np.float32)
    vol_bnds[:, 0] = transformed_pts.min(0)
    vol_bnds[:, 1] = transformed_pts.max(0)
    vol_bnds[2] = [-0.8, 0]

    coordinate_frames = []
    for i, T in enumerate(T_list):
        T = extrinsic @ T
        T_list[i] = T
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coord.transform(T)
        coordinate_frames.append(coord)

    vol_box = o3d.geometry.OrientedBoundingBox()
    vol_box.center = vol_bnds.mean(1)
    vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
    vol_box.color = [1, 0, 0]
    o3d.visualization.draw_geometries(coordinate_frames + [vol_box])

    cfg = dict()
    cfg['tsdf_voxel_size'] = 0.003  # in meter
    cfg['tsdf_trunc_margin'] = 0.015  # in meter
    cfg['pcd_voxel_size'] = 0.005  # in meter

    tsdf = TSDFVolume(vol_bnds=vol_bnds,
                      voxel_size=cfg['tsdf_voxel_size'],
                      trunc_margin=cfg['tsdf_trunc_margin'])

    vol_box = o3d.geometry.OrientedBoundingBox()
    vol_box.center = vol_bnds.mean(1)
    vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
    vol_box.color = [1, 0, 0]
    o3d.visualization.draw_geometries(coordinate_frames + [vol_box])

    for i in tqdm(range(len(T_list))):
        T = T_list[i]
        depth_file = depth_files[i]
        prefix = os.path.basename(depth_file)[:-4]
        color_im_path = os.path.join(args.dataset, f"images/{prefix}.jpg")
        depth_im_path = os.path.join(args.dataset, f"depth/images/{prefix}.png")
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        tsdf.integrate(color_im, depth_im, cam_intr, T)

    opt_recon = tsdf.get_surface_cloud_marching_cubes(voxel_size=0.005)
    o3d.visualization.draw_geometries([vol_box, opt_recon])
