import open3d as o3d
import numpy as np
import scipy.linalg as la
from skimage import measure

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray

from .cuda_kernels import source_module


class TSDFVolume:

    def __init__(self, voxel_size, vol_bnds=None, vol_origin=None, vol_dim=None, trunc_margin=0.015):
        """
        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        if vol_dim is not None and vol_origin is not None:
            self._vol_dim = vol_dim
            self._vol_origin = vol_origin
        elif vol_bnds is not None:
            self._vol_bnds = np.ascontiguousarray(vol_bnds, dtype=np.float32)
            self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / voxel_size).astype(np.int32)
            self._vol_origin = self._vol_bnds[:, 0].copy()
        else:
            raise ValueError("must either provide 'vol_dim' or (both 'vol_bnds' and 'vol_origin')")

        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin
        self._color_const = np.float32(256 * 256)

        print(f"TSDF volume dim: {self._vol_dim}, # points: {np.prod(self._vol_dim)}")

        # Copy voxel volumes to GPU
        self.block_x, self.block_y, self.block_z = 8, 8, 16  # block_x * block_y * block_z must equal to 1024

        x_dim, y_dim, z_dim = int(self._vol_dim[0]), int(self._vol_dim[1]), int(self._vol_dim[2])
        self.grid_x = int(np.ceil(x_dim / self.block_x))
        self.grid_y = int(np.ceil(y_dim / self.block_y))
        self.grid_z = int(np.ceil(z_dim / self.block_z))

        # initialize tsdf values to be -1
        xyz = x_dim * y_dim * z_dim
        self._tsdf_vol_gpu = gpuarray.zeros(shape=(xyz), dtype=np.float32) - 1
        self._weight_vol_gpu = gpuarray.zeros(shape=(xyz), dtype=np.float32)
        self._color_vol_gpu= gpuarray.zeros(shape=(xyz), dtype=np.float32)

        # integrate function using PyCuda
        self._cuda_integrate = source_module.get_function("integrate")
        self._cuda_ray_casting = source_module.get_function("rayCasting")
        self._cuda_batch_ray_casting = source_module.get_function("batchRayCasting")

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, weight=1.0):
        """ Integrate an RGB-D frame into the TSDF volume.

        Args:
            color_im (np.ndarray): input RGB image of shape (H, W, 3)
            depth_im (np.ndarray): input depth image of shape (H, W)
            cam_intr (np.ndarray): Camera intrinsics matrix of shape (3, 3)
            cam_pose (np.ndarray): Camera pose of shape (4, 4)
            weight (float, optional): weight to be assigned for the current observation. Defaults to 1.0.
        """
        im_h, im_w = depth_im.shape

        # color image is always from host
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

        if isinstance(depth_im, np.ndarray):
            depth_im_gpu = gpuarray.to_gpu(depth_im.astype(np.float32))

        self._cuda_integrate(
            self._tsdf_vol_gpu,
            self._weight_vol_gpu,
            self._color_vol_gpu,
            cuda.InOut(self._vol_dim.astype(np.int32)),
            cuda.InOut(self._vol_origin.astype(np.float32)),
            np.float32(self._voxel_size),
            cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
            np.int32(im_h),
            np.int32(im_w),
            cuda.InOut(color_im.astype(np.float32)),
            depth_im_gpu,
            np.float32(self._trunc_margin),
            np.float32(weight),
            block=(self.block_x, self.block_y, self.block_z),
            grid=(self.grid_x, self.grid_y, self.grid_z)
        )

    def batch_ray_casting(self, im_w, im_h, cam_intr, cam_poses, inv_cam_poses, start_row=0, start_col=0,
                          batch_size=1, to_host=True):
        batch_color_im_gpu = gpuarray.zeros(shape=(batch_size, 3, im_h, im_w), dtype=np.uint8)
        batch_depth_im_gpu = gpuarray.zeros(shape=(batch_size, im_h, im_w), dtype=np.float32)

        self._cuda_batch_ray_casting(
            self._tsdf_vol_gpu,
            self._color_vol_gpu,
            self._weight_vol_gpu,
            cuda.InOut(self._vol_dim.astype(np.int32)),
            cuda.InOut(self._vol_origin.astype(np.float32)),
            np.float32(self._voxel_size),
            cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
            cuda.InOut(cam_poses.reshape(-1).astype(np.float32)),
            cuda.InOut(inv_cam_poses.reshape(-1).astype(np.float32)),
            np.int32(start_row),
            np.int32(start_col),
            np.int32(im_h),
            np.int32(im_w),
            batch_color_im_gpu,
            batch_depth_im_gpu,
            np.int32(batch_size),
            block=(8, 8, 16),
            grid=(int(np.ceil(im_w / 8)), int(np.ceil(im_h / 8)), int(np.ceil(batch_size / 16)))
        )
        if not to_host:
            return batch_depth_im_gpu, batch_color_im_gpu

        batch_depth_im = batch_depth_im_gpu.get()
        batch_color_im = batch_color_im_gpu.get().transpose(0, 2, 3, 1)  # b, h, w, c
        return batch_depth_im, batch_color_im

    def ray_casting(self, im_w, im_h, cam_intr, cam_pose, start_row=0, start_col=0, to_host=True):
        """
        Render an image patch
        """
        depth_im_gpu = gpuarray.zeros(shape=(im_h, im_w), dtype=np.float32)
        color_im_gpu = gpuarray.zeros(shape=(3, im_h, im_w), dtype=np.uint8)

        self._cuda_ray_casting(
            self._tsdf_vol_gpu,
            self._color_vol_gpu,
            self._weight_vol_gpu,
            cuda.InOut(self._vol_dim.astype(np.int32)),
            cuda.InOut(self._vol_origin.astype(np.float32)),
            np.float32(self._voxel_size),
            cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
            cuda.InOut(la.inv(cam_pose).reshape(-1).astype(np.float32)),
            np.int32(start_row),
            np.int32(start_col),
            np.int32(im_h),
            np.int32(im_w),
            color_im_gpu,
            depth_im_gpu,
            block=(32, 32, 1),
            grid=(int(np.ceil(im_w / 32)), int(np.ceil(im_h / 32)), 1)
        )
        if not to_host:
            return depth_im_gpu, color_im_gpu

        depth_im = depth_im_gpu.get()
        color_im = color_im_gpu.get().transpose(1, 2, 0)
        return depth_im, color_im

    def get_volume(self):
        x_dim, y_dim, z_dim = self._vol_dim
        tsdf_vol_cpu = self._tsdf_vol_gpu.get().reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        color_vol_cpu = self._color_vol_gpu.get().reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        weight_vol_cpu = self._weight_vol_gpu.get().reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        return tsdf_vol_cpu, color_vol_cpu, weight_vol_cpu

    def get_surface_cloud_marching_cubes(self, voxel_size=0.005):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        # Marching cubes
        verts, faces, normals, values = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)

        # remove false surface
        verts_weight = weight_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        verts_val = tsdf_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        valid_idx = (verts_weight > 0) & (np.abs(verts_val) < 0.2)
        verts_ind = verts_ind[valid_idx]
        verts = verts[valid_idx]
        normals = normals[valid_idx]

        # make normals point to the inward (positive -> negative) direction
        back_verts = verts - normals
        forward_verts = verts + normals
        back_verts = np.clip(back_verts, a_min=np.zeros(3), a_max=np.array(tsdf_vol.shape)-1)
        forward_verts = np.clip(forward_verts, a_min=np.zeros(3), a_max=np.array(tsdf_vol.shape)-1)
        back_ind = np.round(back_verts).astype(int)
        forward_ind = np.round(forward_verts).astype(int)

        back_val = tsdf_vol[back_ind[:, 0], back_ind[:, 1], back_ind[:, 2]]
        forward_val = tsdf_vol[forward_ind[:, 0], forward_ind[:, 1], forward_ind[:, 2]]
        normals[(forward_val - back_val) > 0] *= -1

        verts = verts*self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
        colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T

        surface_cloud = o3d.geometry.PointCloud()
        surface_cloud.points = o3d.utility.Vector3dVector(verts)
        surface_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
        surface_cloud.normals = o3d.utility.Vector3dVector(normals)
        surface_cloud = surface_cloud.voxel_down_sample(voxel_size=voxel_size)
        return surface_cloud

    def get_conservative_volume(self, voxel_size=0.01):
        tsdf_vol, _, _ = self.get_volume()
        verts = np.vstack(np.where(tsdf_vol < -0.2)).T
        verts = verts*self._voxel_size + self._vol_origin

        conservative_volume = o3d.geometry.PointCloud()
        conservative_volume.points = o3d.utility.Vector3dVector(verts)
        conservative_volume = conservative_volume.voxel_down_sample(voxel_size=voxel_size)
        return conservative_volume

    def save(self, output_path):
        np.savez_compressed(output_path,
            vol_dim=self._vol_dim,
            vol_origin=self._vol_origin,
            voxel_size=self._voxel_size,
            trunc_margin=self._trunc_margin,
            tsdf_vol=self._tsdf_vol_gpu.get(),
            weight_vol=self._weight_vol_gpu.get(),
            color_vol=self._color_vol_gpu.get()
        )
        print(f"tsdf volume has been saved to: {output_path}")

    @classmethod
    def load(cls, input_path):
        loaded = np.load(input_path)
        print('loaded voxel_size:', loaded['voxel_size'])
        print('loaded vol_bnds', loaded.get('vol_bounds'))
        print('loaded vol_origin', loaded.get('vol_origin'))
        print('loaded vol_dim:', loaded.get('vol_dim'))
        print('loaded trunc_margin:', loaded['trunc_margin'])
        obj = cls(voxel_size=loaded['voxel_size'], vol_bnds=loaded.get('vol_bounds'),
                  vol_dim=loaded.get('vol_dim'), vol_origin=loaded.get('vol_origin'),
                  trunc_margin=loaded['trunc_margin'])
        obj._tsdf_vol_gpu = gpuarray.to_gpu(loaded['tsdf_vol'])
        obj._weight_vol_gpu = gpuarray.to_gpu(loaded['weight_vol'])
        obj._color_vol_gpu = gpuarray.to_gpu(loaded['color_vol'])
        print(f"tsdf volume has been loaded from: {input_path}")
        return obj
