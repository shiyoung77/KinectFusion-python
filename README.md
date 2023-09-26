# KinectFusion-python
A basic implementation of KinectFusion \[1\] in Python inspired by https://github.com/andyzeng/tsdf-fusion-python. \
It's light and fast (25~30 FPS with my RTX3090). \
It has been tested on tabletop scenes for robot manipulation and small rooms for navigation :)

## Dependencies
- Python >=3.6
- [CuPy](https://cupy.dev/) (check the installation guide on the official website)
- [Open3D](https://github.com/isl-org/Open3D) (pip install open3d)
- [cupoch](https://github.com/neka-nat/cupoch) (pip install cupoch)

## Usage
```
git clone git@github.com:shiyoung77/KinectFusion-python.git
python -m KinectFusion-python.main --dataset {dataset_path} --video {video_name}
```
If you want to save not only the poses and reconstructed point cloud but also the TSDF volume, add `--save_tsdf` option. \
By default it saves the result in the video folder. You can specify the output directory with `--output_dir` option. \
Check `main.py` for more command line options.
Check the `kf_config.py` for the default parameters for reconstruction.

## Prepare your own dataset
You should have your dataset in the following format.
```
{dataset_path}/
    {video_name}/
        color/
            0000-color.png(or jpg)
            0001-color.png
            ...
        depth/
            0000-depth.png
            0001-depth.png
            ...
        config.json
    {second_video_name}/
        ...
```
config.json should contain the camera information. An example config.json is as follows.
```
{
    "id": "video0",
    "im_w": 640,
    "im_h": 480,
    "depth_scale": 1000,
    "cam_intr": [
        [
            1066.778,
            0,
            312.9869
        ],
        [
            0,
            1067.487,
            241.3109
        ],
        [
            0,
            0,
            1
        ]
    ]
}
```
An example RGB-D video capture at [Rutgers CS Robotics Lab](https://robotics.cs.rutgers.edu) could be found at this [google drive link](https://drive.google.com/file/d/1TGbuWPAaXomunjj0my0wNbpDLSUg4DEK/view?usp=sharing).

## Reference
\[1\] Newcombe, R. A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A. J., ... & Fitzgibbon, A. (2011, October). Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality (pp. 127-136). IEEE.
