import os
import copy
from pathlib import Path

import numpy as np
import numpy.linalg as la
import open3d as o3d
import cupoch as cph
from cupoch import registration as reg

from .tsdf_lib import TSDFVolume
from . import kf_utils as utils


class KinectFusion:

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.tsdf_volume = None
        self.init_transformation = None
        self.transformation = None  # model transformation, i.e. la.inv(cam_pose)
        self.prev_pcd = None
        self.cam_poses = []  # store the tracking results
        self.vol_box = None

    def initialize_tsdf_volume(self, color_im, depth_im, visualize=False):
        pcd = utils.create_pcd(depth_im, self.cfg['cam_intr'], color_im, depth_trunc=3)

        plane_frame, inlier_ratio = utils.plane_detection_o3d(pcd,
                                                              max_iterations=1000,
                                                              inlier_thresh=0.005,
                                                              visualize=False)
        cam_pose = la.inv(plane_frame)
        transformed_pcd = copy.deepcopy(pcd).transform(la.inv(plane_frame))
        transformed_pts = np.array(transformed_pcd.points)
        transformed_pts = transformed_pts[transformed_pts[:, 2] > -0.05]

        vol_bnds = np.zeros((3, 2), dtype=np.float32)
        vol_bnds[:, 0] = transformed_pts.min(0)
        vol_bnds[:, 1] = transformed_pts.max(0)
        vol_bnds[0] += self.cfg['bound_dx']
        vol_bnds[1] += self.cfg['bound_dy']
        vol_bnds[2] = self.cfg['bound_z']

        if visualize:
            vol_box = o3d.geometry.OrientedBoundingBox()
            vol_box.center = vol_bnds.mean(1)
            vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
            vol_box.color = [1, 0, 0]
            self.vol_box = vol_box
            cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
            world_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
            world_frame.transform(cam_pose)
            o3d.visualization.draw_geometries([vol_box, transformed_pcd, world_frame, cam_frame])

        self.init_transformation = plane_frame.copy()
        self.transformation = plane_frame.copy()
        self.tsdf_volume = TSDFVolume(vol_bnds=vol_bnds,
                                      voxel_size=self.cfg['tsdf_voxel_size'],
                                      trunc_margin=self.cfg['tsdf_trunc_margin'])
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose)
        self.prev_pcd = utils.create_pcd_cph(depth_im, self.cfg['cam_intr'], color_im)
        self.cam_poses.append(cam_pose)

    @staticmethod
    def multiscale_icp(src: cph.geometry.PointCloud,
                       tgt: cph.geometry.PointCloud,
                       voxel_size_list: list,
                       max_iter_list: list,
                       init: np.ndarray = np.eye(4),
                       inverse: bool = False):

        if len(src.points) > len(tgt.points):
            return KinectFusion.multiscale_icp(tgt, src, voxel_size_list, max_iter_list,
                                               init=la.inv(init), inverse=True)

        assert np.all(sorted(voxel_size_list, key=lambda x: -x) == voxel_size_list),\
            "voxel_size_list is not in descending order"

        src = src.voxel_down_sample(voxel_size_list[-1])
        tgt = tgt.voxel_down_sample(voxel_size_list[-1])
        src.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))
        tgt.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))

        transformation = init.astype(np.float32)
        result_icp = None
        for i, (voxel_size, max_iter) in enumerate(zip(voxel_size_list, max_iter_list)):
            if i != len(voxel_size_list) - 1:
                src_down = src.voxel_down_sample(voxel_size)
                tgt_down = src.voxel_down_sample(voxel_size)
            else:
                src_down = src
                tgt_down = tgt

            result_icp = reg.registration_icp(
                src_down, tgt_down,
                max_correspondence_distance=voxel_size * 3,
                init=transformation,
                estimation_method=reg.TransformationEstimationPointToPlane(),
                criteria=reg.ICPConvergenceCriteria(max_iteration=max_iter)
            )
            transformation = result_icp.transformation

        if inverse and result_icp is not None:
            result_icp.transformation = la.inv(result_icp.transformation)

        return result_icp

    def compute_model_to_frame_transformation(self, curr_pcd):
        cam_pose = la.inv(self.transformation)
        rendered_depth, rendered_color = self.tsdf_volume.ray_casting(self.cfg['im_w'],
                                                                      self.cfg['im_h'],
                                                                      self.cfg['cam_intr'],
                                                                      cam_pose, to_host=True)
        rendered_pcd = utils.create_pcd_cph(rendered_depth, self.cfg['cam_intr'], rendered_color)
        result_icp = self.multiscale_icp(rendered_pcd,
                                         curr_pcd,
                                         voxel_size_list=[0.025, 0.01],
                                         max_iter_list=[10, 20])
        return result_icp

    def compute_frame_to_frame_transformation(self, curr_pcd):
        cam_intr = cph.camera.PinholeCameraIntrinsic()
        cam_intr.intrinsic_matrix = self.cfg['cam_intr']

        result_icp = self.multiscale_icp(self.prev_pcd,
                                         curr_pcd,
                                         voxel_size_list=[0.025, 0.01],
                                         max_iter_list=[10, 20])
        return result_icp

    def update(self, color_im, depth_im):
        assert self.tsdf_volume is not None, "TSDF volume has not been initialized."

        # get current point cloud
        curr_pcd = utils.create_pcd_cph(depth_im, self.cfg['cam_intr'], color_im)
        result_icp = self.compute_model_to_frame_transformation(curr_pcd)
        delta_T = result_icp.transformation
        delta_R = delta_T[:3, :3]
        delta_t = delta_T[:3, 3]

        # sanity check
        translation_distance = la.norm(delta_t)
        factor = np.clip((np.trace(delta_R) - 1) / 2, a_min=-1, a_max=1)
        rotation_distance = np.arccos(factor)
        if translation_distance > 0.1 or rotation_distance > np.pi / 6:
            print("Sanity check fail, no integration.")
            self.cam_poses.append(np.full_like(self.cam_poses[-1], np.nan))
            return False

        self.transformation = delta_T @ self.transformation
        cam_pose = la.inv(self.transformation)
        self.cam_poses.append(cam_pose)
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose, weight=1)
        self.prev_pcd = curr_pcd
        return True

    def save(self, output_folder, save_tsdf=False):
        output_folder = Path(output_folder)
        output_path = output_folder / 'kf_results.npz'
        if output_path.exists():
            key = input(f"{output_path} exists. Do you want to overwrite? (y/n)")
            while key.lower() not in ['y', 'n']:
                key = input(f"{output_path} exists. Do you want to overwrite? (y/n)")
            if key.lower() == 'n':
                return
        else:
            output_folder.mkdir(parents=True, exist_ok=True)
        cam_poses = np.stack(self.cam_poses)
        np.savez_compressed(output_path, cam_poses=cam_poses, **self.cfg)

        output_path = output_folder / 'scan.pcd'
        surface = self.tsdf_volume.get_surface_cloud_marching_cubes()
        o3d.io.write_point_cloud(str(output_path), surface)

        if save_tsdf:
            output_path = output_folder / 'tsdf.npz'
            self.tsdf_volume.save(output_path)
        return surface

    @classmethod
    def load(cls, output_folder):
        kf_results_path = os.path.join(output_folder, 'kf_results.npz')
        tsdf_path = os.path.join(output_folder, 'tsdf.npz')
        assert os.path.exists(tsdf_path), f"TSDF file is not found at: {tsdf_path}"
        assert os.path.exists(kf_results_path), f"Fusion result file is not found at: {kf_results_path}"

        kf_results = dict()
        cam_poses = None
        with np.load(kf_results_path) as data:
            for key in data.files:
                if key == "cam_poses":
                    cam_poses = data["cam_poses"]
                else:
                    kf_results[key] = data[key]
        kf = cls(cfg=kf_results)
        kf.tsdf_volume = TSDFVolume.load(tsdf_path)
        kf.cam_poses = cam_poses
        kf.transformation = la.inv(cam_poses[-1])
        return kf
