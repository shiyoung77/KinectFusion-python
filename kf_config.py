import numpy as np


def get_config():
    config = dict()
    config['tsdf_voxel_size'] = 0.003  # in meter
    config['tsdf_trunc_margin'] = 0.015  # in meter
    config['pcd_voxel_size'] = 0.005  # in meter
    config['bound_dx'] = [-0.2, 0.2]  # in meter
    config['bound_dy'] = [-0.2, 0.4]  # in meter
    config['bound_z'] = [-0.1, 0.7]  # in meter
    return config


def print_config(cfg):
    for key, value in cfg.items():
        if isinstance(value, np.ndarray):
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    print_config(get_config())
