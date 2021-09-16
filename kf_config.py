import numpy as np

def get_config():
    config = dict()
    config['tsdf_voxel_size'] = 0.002  # in meter
    config['tsdf_trunc_margin'] = 0.015 # in meter
    config['pcd_voxel_size'] = 0.005  # in meter
    return config

def print_config(cfg):
    for key, value in cfg.items():
        if isinstance(value, np.ndarray):
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    cfg = get_config()
    print_config(cfg)
