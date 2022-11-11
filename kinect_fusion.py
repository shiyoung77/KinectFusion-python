import os
import copy
from time import perf_counter

import numpy as np
import scipy.linalg as la
import cv2
import open3d as o3d
import cupoch as cph
from cupoch import registration as reg

from .tsdf_lib import TSDFVolume
from . import kf_utils as utils
from .feature_matching import extract_sift_keypoints, matching_sift_keypoints


class KinectFusion:

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        self.tsdf_volume = None
        self.opt_tsdf_volume = None
        self.init_transformation = None
        self.transformation = None  # model transformation, i.e. la.inv(cam_pose)
        self.prev_pcd = None
        self.cam_poses = []  # store the tracking results
        self.valid_pose_indices = []
        self.vol_box = None
        self.keypoints = []
        self.descriptions = []
        self.count = 0
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def initialize_tsdf_volume(self, color_im, depth_im, visualize=False):
        pcd = utils.create_pcd(depth_im, self.cfg['cam_intr'], color_im, depth_trunc=3)

        plane_frame, inlier_ratio = utils.timeit(utils.plane_detection_o3d)(pcd,
                                                                            max_iterations=500,
                                                                            inlier_thresh=0.005,
                                                                            visualize=False)
        cam_pose = la.inv(plane_frame)
        transformed_pcd = copy.deepcopy(pcd).transform(la.inv(plane_frame))
        transformed_pts = np.array(transformed_pcd.points)
        transformed_pts = transformed_pts[transformed_pts[:, 2] > -0.05]

        vol_bnds = np.zeros((3, 2), dtype=np.float32)
        vol_bnds[:, 0] = transformed_pts.min(0) - 1
        vol_bnds[:, 1] = transformed_pts.max(0) + 1
        vol_bnds[2] = [-0.2, 0.5]

        if visualize:
            vol_box = o3d.geometry.OrientedBoundingBox()
            vol_box.center = vol_bnds.mean(1)
            vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
            vol_box.color = [1, 0, 0]
            self.vol_box = vol_box
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            world_frame.transform(cam_pose)
            o3d.visualization.draw_geometries([vol_box, transformed_pcd, world_frame, cam_frame])

        self.init_transformation = plane_frame.copy()
        self.transformation = plane_frame.copy()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(cam_pose))

        self.tsdf_volume = TSDFVolume(vol_bnds=vol_bnds,
                                      voxel_size=self.cfg['tsdf_voxel_size'],
                                      trunc_margin=self.cfg['tsdf_trunc_margin'])
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose)
        # self.opt_tsdf_volume = TSDFVolume(vol_bnds=vol_bnds,
        #                                   voxel_size=self.cfg['tsdf_voxel_size'],
        #                                   trunc_margin=self.cfg['tsdf_trunc_margin'])

        self.prev_pcd = utils.create_pcd_cph(depth_im, self.cfg['cam_intr'], color_im)
        self.cam_poses.append(cam_pose)
        self.valid_pose_indices.append(self.count)

        # extract SIFT keypoints and their descriptions (features)
        H, W, _ = color_im.shape
        resized_color_im = cv2.resize(color_im, (W//2, H//2), cv2.INTER_CUBIC)
        keypoints, descriptions = extract_sift_keypoints(resized_color_im)
        self.keypoints.append(keypoints)
        self.descriptions.append(descriptions)

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
                src_down, tgt_down, max_correspondence_distance=voxel_size * 3,
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

    def optimize_pose_graph(self):
        print("Optimizing PoseGraph ...")
        tic = perf_counter()
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.02,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option
            )
        print(f"Graph optimization takes {perf_counter() - tic}s.")

    def update_pose(self, color_im, depth_im):
        self.count += 1

        # get current point cloud
        curr_pcd = utils.create_pcd_cph(depth_im, self.cfg['cam_intr'], color_im)

        # result_icp = self.compute_frame_to_frame_transformation(curr_pcd)
        result_icp = self.compute_model_to_frame_transformation(curr_pcd)
        if result_icp is None:
            return False

        delta_T = result_icp.transformation
        delta_R = delta_T[:3, :3]
        delta_t = delta_T[:3, 3]

        self.transformation = delta_T @ self.transformation
        cam_pose = la.inv(self.transformation)
        self.cam_poses.append(cam_pose)
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose, weight=1)

        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(cam_pose))
        self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(self.count - 1,
                                                                              self.count,
                                                                              transformation=delta_T,
                                                                              information=np.eye(6),
                                                                              uncertain=False))

        '''
        # extract SIFT keypoints and their descriptions (features)
        keypoints, descriptions = extract_sift_keypoints(color_im)
        self.keypoints.append(keypoints)
        self.descriptions.append(descriptions)

        chunk_size = 10
        max_neighbor_chunk = 3
        if self.count % chunk_size == 0:
            # add intra-chunk edges
            self.dense_matching(chunk_size)

            # add inter-chunk edges
            chunk_idx = self.count // chunk_size
            for i in range(2, max_neighbor_chunk + 1):
                neighbor_chunk_idx = chunk_idx - i
                if neighbor_chunk_idx <= 0:
                    break
                start_id = neighbor_chunk_idx * chunk_size
                end_id = chunk_idx * chunk_size
                self.sparse_matching(start_id, end_id)
        '''

        self.prev_pcd = curr_pcd

    def sparse_matching(self, i, j):
        cam_intr = np.array(self.cfg['cam_intr'])
        kp1 = self.keypoints[i]
        des1 = self.descriptions[i]
        kp2 = self.keypoints[j]
        des2 = self.descriptions[j]
        R = matching_sift_keypoints(kp1, des1, kp2, des2, cam_intr)[0]
        pose = np.eye(4)
        pose[:3, :3] = R
        information = np.diag([1, 1, 1, 0, 0, 0])
        edge = o3d.pipelines.registration.PoseGraphEdge(i, j,
                                                        transformation=pose,
                                                        information=information,
                                                        uncertain=True,
                                                        confidence=0.5)
        self.pose_graph.edges.append(edge)

    def dense_matching(self, chunk_size):
        cam_intr = np.array(self.cfg['cam_intr'])
        kp_list = self.keypoints[-chunk_size - 1:]
        des_list = self.descriptions[-chunk_size - 1:]
        start_id = self.count - chunk_size
        for i in range(chunk_size + 1):
            for j in range(i + 1, chunk_size + 1):
                R = matching_sift_keypoints(kp_list[i], des_list[i], kp_list[j], des_list[j], cam_intr)[0]
                if np.allclose(R, np.eye(3)):
                    continue

                pose = np.eye(4)
                pose[:3, :3] = R
                information = np.diag([1, 1, 1, 0, 0, 0])
                edge = o3d.pipelines.registration.PoseGraphEdge(i + start_id,
                                                                j + start_id,
                                                                transformation=pose,
                                                                information=information,
                                                                uncertain=True,
                                                                confidence=0.5)
                self.pose_graph.edges.append(edge)

    def update(self, color_im, depth_im):
        assert self.tsdf_volume is not None, "TSDF volume has not been initialized."
        self.count += 1

        # get current point cloud
        curr_pcd = utils.create_pcd_cph(depth_im, self.cfg['cam_intr'], color_im)
        result_icp = self.compute_model_to_frame_transformation(curr_pcd)
        if result_icp is None:
            return False

        delta_T = result_icp.transformation
        delta_R = delta_T[:3, :3]
        delta_t = delta_T[:3, 3]

        # sanity check
        translation_distance = la.norm(delta_t)
        rotation_distance = np.arccos((np.trace(delta_R) - 1) / 2)
        if translation_distance > 0.1 or rotation_distance > np.pi / 6:
            print("Sanity check fail, no integration.")
            return False

        self.transformation = delta_T @ self.transformation
        cam_pose = la.inv(self.transformation)
        self.cam_poses.append(cam_pose)
        self.tsdf_volume.integrate(color_im, depth_im, self.cfg['cam_intr'], cam_pose, weight=1)
        self.valid_pose_indices.append(self.count)
        self.prev_pcd = curr_pcd

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
        return surface
