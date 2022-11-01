import os
import time
import copy
import open3d as o3d
import open3d.core as o3c
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
        self.cam_intr = np.asarray(self.cfg['cam_intr'])
        self.cam_intr_t = o3c.Tensor(self.cam_intr)

    def initialize_tsdf_volume(self, color_im, depth_im, visualize=False):
        pcd = utils.create_pcd(depth_im, self.cam_intr, color_im, depth_trunc=3)
        plane_frame, inlier_ratio = utils.plane_detection_o3d(pcd, max_iterations=500, inlier_thresh=0.005)

        cam_pose = la.inv(plane_frame)
        transformed_pcd = copy.deepcopy(pcd).transform(la.inv(plane_frame))
        transformed_pts = np.asarray(transformed_pcd.points)

        vol_bnds = np.zeros((3, 2), dtype=np.float32)
        vol_bnds[:, 0] = transformed_pts.min(0)
        vol_bnds[:, 1] = transformed_pts.max(0)
        vol_bnds[2] = [-0.01, 0.45]

        if visualize:
            vol_box = o3d.geometry.OrientedBoundingBox()
            vol_box.color = [1, 0, 0]
            vol_box.center = vol_bnds.mean(1)
            vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            o3d.visualization.draw_geometries([vol_box, transformed_pcd, coord])

        self.init_transformation = plane_frame.copy()
        self.transformation = plane_frame.copy()
        self.tsdf_volume = TSDFVolume(vol_bnds=vol_bnds,
                                      voxel_size=self.cfg['tsdf_voxel_size'],
                                      trunc_margin=self.cfg['tsdf_trunc_margin'])
        self.tsdf_volume.integrate(color_im, depth_im, self.cam_intr, cam_pose)

        self.prev_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        self.cam_poses.append(cam_pose)

    # @staticmethod
    # def multiscale_icp(src: cph.geometry.PointCloud,
    #                    tgt: cph.geometry.PointCloud,
    #                    voxel_size_list: list,
    #                    max_iter_list: list,
    #                    init: np.ndarray = np.eye(4),
    #                    inverse: bool = False):

    #     if len(src.points) > len(tgt.points):
    #         return KinectFusion.multiscale_icp(tgt, src, voxel_size_list, max_iter_list,
    #                                            init=la.inv(init), inverse=True)

    #     transformation = init.astype(np.float32)
    #     result_icp = None
    #     for i, (voxel_size, max_iter) in enumerate(zip(voxel_size_list, max_iter_list)):
    #         src_down = src.voxel_down_sample(voxel_size)
    #         tgt_down = tgt.voxel_down_sample(voxel_size)

    #         src_down.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))
    #         tgt_down.estimate_normals(cph.geometry.KDTreeSearchParamKNN(knn=30))

    #         result_icp = reg.registration_icp(
    #             src_down, tgt_down, max_correspondence_distance=voxel_size*3,
    #             init=transformation,
    #             estimation_method=reg.TransformationEstimationPointToPlane(),
    #             criteria=reg.ICPConvergenceCriteria(max_iteration=max_iter)
    #         )
    #         transformation = result_icp.transformation

    #     if inverse and result_icp is not None:
    #         result_icp.transformation = la.inv(result_icp.transformation)

    #     return result_icp

    def update_pose_using_icp(self, depth_im):
        im_h, im_w = depth_im.shape
        curr_pcd = utils.create_pcd(depth_im, cam_intr=self.cam_intr, depth_trunc=3)
        curr_pcd_t = o3d.t.geometry.PointCloud.from_legacy(curr_pcd)

        cam_pose = la.inv(self.transformation)
        rendered_depth, _ = self.tsdf_volume.ray_casting(im_w, im_h, self.cam_intr, cam_pose, to_host=True)

        rendered_depth = o3d.geometry.Image(rendered_depth)
        rendered_pcd = utils.create_pcd(rendered_depth, cam_intr=self.cam_intr, depth_trunc=1.5)
        rendered_pcd_t = o3d.t.geometry.PointCloud.from_legacy(rendered_pcd)

        rendered_pcd_t = rendered_pcd_t.cuda()
        curr_pcd_t = curr_pcd_t.cuda()

        treg = o3d.t.pipelines.registration
        tic = time.time()
        result_icp = treg.multi_scale_icp(
            rendered_pcd_t,
            curr_pcd_t,
            voxel_sizes=o3d.utility.DoubleVector([0.025, 0.01]),
            criteria_list=[treg.ICPConvergenceCriteria(max_iteration=10) for _ in range(2)],
            max_correspondence_distances=o3d.utility.DoubleVector([0.075, 0.03]),
            init_source_to_target=o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32),
            estimation_method=treg.TransformationEstimationPointToPoint(),
        )
        # print(time.time() - tic)

        delta_transformation = result_icp.transformation.numpy()

        if result_icp is None:
            return False
        
        self.transformation = delta_transformation @ self.transformation
        self.prev_pcd = curr_pcd_t
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

