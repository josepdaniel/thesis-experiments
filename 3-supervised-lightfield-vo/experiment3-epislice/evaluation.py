import sys
sys.path.insert(0, "../")

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
import os
import time
from model import OdometryNet
from epi.loader import EpiDataset, EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip
from epi.utils import RECTIFIED_IMAGE_SHAPE
from configuration import get_config
import matplotlib.pyplot as plt
    

""" Get the relative poses for each frame-pair """
def get_actual_poses(force_recalculate=False):
    already_calculated = os.path.exists("./results/predictions.npy")

    if force_recalculate or not already_calculated:
        cfg = get_config()
        predict_trajectory_every_frame(cfg)

    poses = np.load("./results/actual.npy")
    return poses




""" Get relative poses for each predicted frame-pair """
def get_predicted_poses(force_recalculate=False, averaging=np.nanmean):
    already_calculated = os.path.exists("./results/predictions.npy")

    if force_recalculate or not already_calculated:
        cfg = get_config()
        predict_trajectory_every_frame(cfg)

    predicted = np.load("./results/predictions.npy")
    predicted = averaging(predicted, axis=1)
    return predicted




""" Compute the trajectory based on actual measurements from the arm """
def get_actual_trajectory(force_recalculate=False):
    already_calculated = os.path.exists("./results/actual.npy")

    if force_recalculate or not already_calculated:
        cfg = get_config()
        predict_trajectory_every_frame(cfg)

    trajectory = np.load("./results/actual.npy")
    trajectory = trajectory_from_poses(trajectory)
    origin = np.array([[0,0,0]])
    trajectory = np.concatenate([origin, trajectory], axis=0)

    return trajectory



""" Compute the trajectory based on predicted odometry """
def get_predicted_trajectory(force_recalculate=False, averaging=np.nanmean):
    already_calculated = os.path.exists("./results/predictions.npy")

    if force_recalculate or not already_calculated:
        cfg = get_config()
        predict_trajectory_every_frame(cfg)

    predicted = np.load("./results/predictions.npy")
    predicted = averaging(predicted, axis=1)
    trajectory = trajectory_from_poses(predicted)
    origin = np.array([[0,0,0]])
    trajectory = np.concatenate([origin, trajectory], axis=0)
    
    return trajectory



""" Compute a trajectory from a set of poses - CHEAP version"""
def trajectory_from_poses(p):
	cumx, cumy, cumz = 0, 0, 0
	cumx_pred, cumy_pred, cumz_pred = 0, 0, 0

	absolute_poses = np.zeros_like(p)[:, :3]

	for i in range(0, p.shape[0]):
		row = p[i, :]
		rx, ry, rz, x, y, z = row
		cumx += x 
		cumy += y
		cumz += z
		absolute_poses[i, :] = [cumx, cumy, cumz]

	return absolute_poses
		

""" 
Predict the trajectory for every frame given a model configuration. Since each frame is seen
multiple times, each pose pair can have up to 6 different predictions. This function stores
all available predictions for each pose pair, as a tensor that looks like:
                                    
                                    |   P - - -   |
                                    |   P P - -   |
                                    |   P P P -   |
                                    |   P P P P   |
                                    |   P P P P   |
                                    |   - P P P   |
                                    |   - - P P   |
                                    |   - - - P   |

"""
def predict_trajectory_every_frame(cfg):

    device = "cuda" if torch.cuda.is_available else "cpu"

    ds_options = cfg.ds_options
    ds_options.backwards = False

    N, H, W, C = RECTIFIED_IMAGE_SHAPE
    img_width =  int(W * ds_options.image_scale)
    img_height = int(H * ds_options.image_scale)
    img_channels = len(ds_options.camera_array_indices)*3

    model = OdometryNet(img_channels, img_height, img_width, batchNorm=True)

    checkpoint = torch.load(cfg.save_name)
    weights = checkpoint["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    preprocessing = cfg.preprocessing 

    ds = EpiDataset(
        "/home/joseph/Documents/epidata/smooth/valid/", 
        preprocessing=preprocessing,
        augmentation=None, 
        options=ds_options, 
        sequences=["thegang3"]
    )

    actual_poses = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    # Numbers of images per window
    window_length = ds_options.sequence_length
    poses_per_window = window_length - 1

    # Number of 7-frame windows in the video
    num_windows = len(ds)

    # Number of frames in the video
    num_frames = num_windows + window_length - 2

    # 6 degrees of freedom
    num_measurements = 6

    # Record all measurements in recorder_array to be averaged later
    recorder_array_shape = [num_frames, num_windows, num_measurements]
    recorder_array = np.empty(recorder_array_shape)
    recorder_array[:,:,:] = np.nan

    # Record ground truth measurements in this empty array
    actual_poses_shape = [num_frames, num_measurements]
    actual_poses = np.empty(actual_poses_shape)

    with torch.no_grad():
        for i, window in enumerate(ds):
            print("Predicting {}/{}".format(i, len(ds)))

            # Get ground truth pose tensor
            # current_pose = torch.FloatTensor([window["poses"][0, :]])
            actual_poses[i:i+poses_per_window, :] = window["poses"]

            # Predict pose and fill recorder tensor
            images = window["images"].unsqueeze(0).to(device)
            prediction = model(images).squeeze().cpu().numpy()
            prediction = prediction.reshape([poses_per_window, num_measurements])
            recorder_array[i:i+poses_per_window, i, :] = prediction

            print(f"{i} {recorder_array[i+5, i, :]}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    np.save("results/predictions.npy", recorder_array)
    np.save("results/actual.npy", actual_poses)


if __name__ == "__main__":

    cfg = get_config()
    # predict_trajectory_every_frame(cfg)
    predicted = get_predicted_trajectory(force_recalculate=False)
    actual = get_actual_trajectory()

    predicted_xs, predicted_ys, predicted_zs = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    actual_xs, actual_ys, actual_zs = actual[:, 0], actual[:, 1], actual[:, 2]

    print(actual.shape)
    print(predicted.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(predicted_xs*2, predicted_zs*1.5)
    ax.plot(actual_xs, actual_zs)
    plt.show()





    

