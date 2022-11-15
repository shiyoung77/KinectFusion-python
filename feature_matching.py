import os
import json
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import cv2
import torch
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
from spsg_models.matching import Matching


def extract_sift_keypoints(img: np.ndarray):
    model = cv2.SIFT_create()
    keypoint, description = model.detectAndCompute(img, None)
    return keypoint, description


def matching_sift_keypoints(kp1, des1, kp2, des2, cam_intr):
    # https://github.com/hsuanhauliu/structure-from-motion-with-OpenCV/blob/master/main.py
    # use flann to perform feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    min_match_count = 10
    if len(good) > min_match_count:
        kp1 = np.array([kp1[m.queryIdx].pt for m in good])
        kp2 = np.array([kp2[m.trainIdx].pt for m in good])

    # https://kornia-tutorials.readthedocs.io/en/latest/image_matching.html
    # E, mask = cv2.findEssentialMat(kp1, kp2, cam_intr, cv2.USAC_MAGSAC, 0.999, 1.0)
    try:
        E, mask = cv2.findEssentialMat(kp1, kp2, cameraMatrix=cam_intr, method=cv2.USAC_MAGSAC, prob=0.5,
                                       threshold=0.999, maxIters=100000)
        # _, R, t, mask = cv2.recoverPose(E, kp1, kp2, cameraMatrix=cam_intr, mask=mask)
        _, R, t, mask = cv2.recoverPose(E, kp1, kp2, cameraMatrix=cam_intr, mask=mask)
    except cv2.error:
        return np.eye(3), np.zeros(3), None, kp1, kp2
    return R, t, mask, kp1, kp2


def feature_matching_sift(img1, img2, cam_intr):
    # https://docs.opencv.org/4.5.5/dc/dc3/tutorial_py_matcher.html
    kp1, des1 = extract_sift_keypoints(img1)
    kp2, des2 = extract_sift_keypoints(img2)
    return matching_sift_keypoints(kp1, des1, kp2, des2, cam_intr)


def feature_matching_loftr(img1, img2, cam_intr, model=None, confidence_thresh=0.95, device="cuda:0"):
    if model is None: 
        model = LoFTR(pretrained="indoor").to(device)

    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255
    img1gray = torch.from_numpy(img1gray).to(torch.float32).reshape(1, 1, *img1gray.shape).to(device)

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255
    img2gray = torch.from_numpy(img2gray).to(torch.float32).reshape(1, 1, *img2gray.shape).to(device)

    input_dict = {"image0": img1gray, "image1": img2gray}
    with torch.inference_mode():
        correspondences = model(input_dict)

    confidences = correspondences['confidence']
    mask = confidences > confidence_thresh
    kp1 = correspondences['keypoints0'][mask].cpu().numpy()
    kp2 = correspondences['keypoints1'][mask].cpu().numpy()

    try:
        # https://kornia-tutorials.readthedocs.io/en/latest/image_matching.html
        E, mask = cv2.findEssentialMat(kp1, kp2, cameraMatrix=cam_intr, method=cv2.USAC_MAGSAC, prob=0.5,
                                       threshold=0.999, maxIters=100000)
        _, R, t, mask = cv2.recoverPose(E, kp1, kp2, cameraMatrix=cam_intr, mask=mask)
    except cv2.error:
        return np.eye(3), np.zeros(3), None, kp1, kp2
    return R, t, mask, kp1, kp2

def feature_extract_and_match_spsg(img0, img1, cam_intr, 
                                   nms_radius = 4, keypoint_threshold = 0.005, max_keypoints = 1024, 
                                   superglue = 'indoor', sinkhorn_iterations = 20, match_threshold = 0.2, 
                                   device = "cuda:0"):
    config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
    }
    
    matching = Matching(config).eval().to(device)

    # Perform the matching.
    inp0 = torch.from_numpy(img0/255.).float()[None, None].to(device)
    inp1 = torch.from_numpy(img1/255.).float()[None, None].to(device)
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    
    try:
    # https://kornia-tutorials.readthedocs.io/en/latest/image_matching.html
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, cameraMatrix=cam_intr, method=cv2.USAC_MAGSAC, prob=0.5,
                                       threshold=0.999, maxIters=100000)
        _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, cameraMatrix=cam_intr, mask=mask)
    except cv2.error:
        return np.eye(3), np.zeros(3), None, mkpts0, mkpts1
    return R, t, mask, mkpts0, mkpts1
    
    


def visualize_matching(img1, img2, mask, kp1, kp2):
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp1]
    keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp2]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(kp1))]
    img_vis = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    plt.imshow(img_vis)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos')
    parser.add_argument("-v", "--video", type=str, default="kitchen_0002")
    parser.add_argument("-i", "--idx1", type=int, default=0)
    parser.add_argument("-j", "--idx2", type=int, default=100)
    args = parser.parse_args()

    data_config_path = os.path.join(args.dataset, args.video, "config.json")
    with open(data_config_path, 'r') as fp:
        data_config = json.load(fp)
    cam_intr = np.asarray(data_config['cam_intr'])

    color_im_path = os.path.join(args.dataset, args.video, 'color', f"{args.idx1:04d}-color.jpg")
    img1 = cv2.imread(color_im_path)

    color_im_path = os.path.join(args.dataset, args.video, 'color', f"{args.idx2:04d}-color.jpg")
    img2 = cv2.imread(color_im_path)

    # img1 = cv2.resize(img1, (320, 240), cv2.INTER_CUBIC)
    # img2 = cv2.resize(img2, (320, 240), cv2.INTER_CUBIC)

    tic = perf_counter()
    R, t, mask, kp1, kp2 = feature_matching_sift(img1, img2, cam_intr)
    print(f"SIFT matching takes {perf_counter() - tic}s")
    print("estimated rotation:")
    print(R)
    visualize_matching(img1, img2, mask, kp1, kp2)

    loftr = LoFTR(pretrained="indoor").cuda()
    tic = perf_counter()
    R, t, mask, kp1, kp2 = feature_matching_loftr(img1, img2, cam_intr, loftr)
    print(f"LoFTR matching takes {perf_counter() - tic}s")
    print("estimated rotation:")
    print(R)
    visualize_matching(img1, img2, mask, kp1, kp2)


if __name__ == "__main__":
    main()
