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
from epi.transforms import euler_to_rotm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_epi import trajectory_from_poses

def predict_trajectory_every_frame(output_file):
    device = "cuda" if torch.cuda.is_available else "cpu"

    ds_options = EpiDatasetOptions()
    ds_options.debug = False
    ds_options.with_pose = True
    ds_options.backwards = False
    ds_options.camera_array_indices = [8]
    ds_options.image_scale = 0.2

    N, H, W, C = RECTIFIED_IMAGE_SHAPE
    img_width =  int(W * ds_options.image_scale)
    img_height = int(H * ds_options.image_scale)

    model = OdometryNet(img_height, img_width, batchNorm=True)
    model.load_state_dict(torch.load("./epi.pth"))
    model.to(device)
    model.eval()

    transforms = Compose([
            Resize(ds_options),
            Normalize(ds_options),
            SelectiveStack(ds_options),
            # RandomHorizontalFlip(ds_options),
            MakeTensor(ds_options)
        ])

    ds = EpiDataset(
        "/home/joseph/Documents/epidata/smooth/valid/", 
        transform=transforms, 
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
    num_frames = num_windows + window_length - 1

    # 6 degrees of freedom
    num_measurements = 6

    # Record all measurements in recorder_array to be averaged later
    recorder_array_shape = [num_frames, num_windows, num_measurements]
    recorder_array = np.empty(recorder_array_shape)
    recorder_array[:,:,:] = np.nan

    with torch.no_grad():
        for i, window in enumerate(ds):
            print("Predicting {}/{}".format(i, len(ds)), end="\r")
            # actual_poses = torch.cat([actual_poses, torch.FloatTensor(x["poses"])])
            images = window["images"].unsqueeze(0).to(device)
            prediction = model(images).squeeze().cpu().numpy()
            prediction = prediction.reshape([poses_per_window, num_measurements])
            recorder_array[i:i+poses_per_window, i, :] = prediction
            print(f"{i} {recorder_array[i+5, i, :]}")


    np.save(output_file, recorder_array)


if __name__ == "__main__":

    PREDICTION_FILE = "./results/experiment2/thegang3-predicted-consensus-aug.npy"
    ACTUAL_TRAJECTORY_FILE = "./results/thegang3-actual.npy"

    if not os.path.exists(PREDICTION_FILE):
        predict_trajectory_every_frame(PREDICTION_FILE)

    predictions = np.load(PREDICTION_FILE)
    actual = np.load(ACTUAL_TRAJECTORY_FILE)

    predictions = np.nanmax(predictions, axis=1)
    # print(predictions)
    trajectory_predicted = trajectory_from_poses(predictions)
    trajectory_actual = trajectory_from_poses(actual)

    xs = trajectory_predicted[:45, 0]
    ys = trajectory_predicted[:45, 1]
    zs = trajectory_predicted[:45, 2]

    plt.plot(xs*1.7, zs*1.5)

    xs = trajectory_actual[:45, 0]
    ys = trajectory_actual[:45, 1]
    zs = trajectory_actual[:45, 2]

    plt.plot(xs, zs)
    plt.show()





    

