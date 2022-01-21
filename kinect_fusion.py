import os
import copy
import open3d as o3d
import cupoch as cph
from cupoch import registration as reg
import numpy as np
import scipy.linalg as la

from .tsdf_lib import TSDFVolume
from . import kf_utils as utils

class KinectFusion:

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.tsdf_volume = None
        self.init_transformation = None
        self.transformation = None
        self.prev_pcd = None
        self.cam_poses = []   # store the tracking results

    def initialize_tsdf_volume(self, color_im, depth_im, visualize=False):
        pcd = utils.create_pcd(depth_im, self.cfg['cam_intr'], color_im, depth_trunc=3)
        # plane_frame, inlier_ratio = utils.timeit(utils.plane_detection_ransac)(pcd, inlier_thresh=0.005,
        #     max_iterations=500, early_stop_thresh=0.4, visualize=True)

        plane_frame, inlier_ratio = utils.timeit(utils.plane_detection_o3d)(pcd,
            max_iterations=500, inlier_thresh=0.005, visualize=False)

        cam_pose = la.inv(plane_frame)
        transformed_pcd = copy.deepcopy(pcd).transform(la.inv(plane_frame))
        transformed_pts = np.asarray(transformed_pcd.points)

        vol_bnds = np.zeros((3, 2), dtype=np.float32)
        vol_bnds[:, 0] = transformed_pts.min(0)
        vol_bnds[:, 1] = transformed_pts.max(0)
        vol_bnds[2] = [-0.01, 0.45]

        if visualize:
            vol_box = o3d.geometry.OrientedBoundingBox()
            vol_box.center = vol_bnds.mean(1)
            vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
            o3d.visualization.draw_geometries([vol_box, transformed_pcd])

        self.init_transformation = plane_frame.copy()
        self.transformation = plane_frame.copy()
        self.tsdf_volume = TSDFVolume(vol_bnds=vol_bnds,
                                      voxel_size=self.cfg['tsdf_voxel_size'],
                                      trunc_margin=self.cfg['tsdf_trunc_margin'])
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose)
        self.prev_pcd = pcd
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

        transformation = init.astype(np.float32)
        result_icp = None
        for i, (voxel_size, max_iter) in enumerate(zip(voxel_size_list, max_iter_list)):
            src_down = src.voxel_down_sample(voxel_size)
            tgt_down = tgt.voxel_down_sample(voxel_size)

            src_down.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))
            tgt_down.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))

            result_icp = reg.registration_icp(
                src_down, tgt_down, max_correspondence_distance=voxel_size*3,
                init=transformation,
                estimation_method=reg.TransformationEstimationPointToPlane(),
                criteria=reg.ICPConvergenceCriteria(max_iteration=max_iter)
            )
            transformation = result_icp.transformation

        if inverse and result_icp is not None:
            result_icp.transformation = la.inv(result_icp.transformation)

        return result_icp

    def update_pose_using_icp(self, depth_im):
        # curr_pcd = utils.create_pcd(depth_im, self.cfg['cam_intr'])

        depth_im= cph.geometry.Image(depth_im)
        cam_intr = cph.camera.PinholeCameraIntrinsic()
        cam_intr.intrinsic_matrix = self.cfg['cam_intr']
        curr_pcd = cph.geometry.PointCloud.create_from_depth_image(depth_im, cam_intr)

        # #------------------------------ frame to frame ICP (open loop) ------------------------------
        # open_loop_fitness = 0
        # result_icp = self.multiscale_icp(self.prev_pcd, curr_pcd,
        #                                  voxel_size_list=[0.025, 0.01, 0.005],
        #                                  max_iter_list=[10, 10, 10], init=np.eye(4))
        # if result_icp is not None:
        #     self.transformation = result_icp.transformation @ self.transformation

        #------------------------------ model to frame ICP (closed loop) ------------------------------
        cam_pose = la.inv(self.transformation)
        rendered_depth, _ = self.tsdf_volume.ray_casting(self.cfg['im_w'], self.cfg['im_h'], self.cfg['cam_intr'],
                                                         cam_pose, to_host=True)
        # rendered_pcd = utils.create_pcd(rendered_depth, self.cfg['cam_intr'])

        rendered_depth = cph.geometry.Image(rendered_depth)
        rendered_pcd = cph.geometry.PointCloud.create_from_depth_image(rendered_depth, cam_intr)
        result_icp = self.multiscale_icp(rendered_pcd,
                                         curr_pcd,
                                         voxel_size_list=[0.025, 0.01],
                                         max_iter_list=[5, 10])
        if result_icp is None:
            return False

        self.transformation = result_icp.transformation @ self.transformation
        self.prev_observation = curr_pcd
        return True

    def update(self, color_im, depth_im):
        assert self.tsdf_volume is not None, "TSDF volume has not been initialized."

        success = self.update_pose_using_icp(depth_im)
        if success:
            cam_pose = la.inv(self.transformation)
            self.cam_poses.append(cam_pose)
            self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose, weight=1)
        else:
            self.cam_poses.append(np.eye(4))

    def save(self, output_folder, voxel_size=0.005):
        if os.path.exists(output_folder):
            key = input(f"{output_folder} exists. Do you want to overwrite? (y/n)")
            while key.lower() not in ['y', 'n']:
                key = input(f"{output_folder} exists. Do you want to overwrite? (y/n)")
            if key.lower() == 'n':
                return
        else:
            os.makedirs(output_folder)

        cam_poses = np.stack(self.cam_poses)
        np.savez_compressed(os.path.join(output_folder, 'kf_results.npz'),
            cam_poses=cam_poses,
            **self.cfg,
        )
        self.tsdf_volume.save(os.path.join(output_folder, 'tsdf.npz'))
        surface = self.tsdf_volume.get_surface_cloud_marching_cubes(voxel_size=voxel_size)
        o3d.io.write_point_cloud(os.path.join(output_folder, 'recon.pcd'), surface)
        print(f"Results have been saved to {output_folder}.")

