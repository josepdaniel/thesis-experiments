import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
import os
from model import OdometryNet
from epi.loader import EpiDataset, EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip
from epi.utils import RECTIFIED_IMAGE_SHAPE
from test_epi import model_test_trajectory
import matplotlib.pyplot as plt

N, H, W, C = RECTIFIED_IMAGE_SHAPE


# --------------- OPTIONS ---------------------------
torch.manual_seed(42)

use_pretrained_flownet = False
pretrained_flownet_path = "./pretrained/flownets_EPE1.951.pth.tar"

resume = True
resume_checkpoint = "/home/joseph/Documents/thesis/2-supervised-stereo-vo/epi.pth"

train_path = "/home/joseph/Documents/epidata/smooth/train"
valid_path = "/home/joseph/Documents/epidata/smooth/valid"

ds_options = EpiDatasetOptions()
ds_options.debug = False
ds_options.with_pose = True
ds_options.camera_array_indices = [8, 9]
ds_options.image_scale = 0.2

epochs = 1000
batch_size = 4
plot_trajectory_every = 50
# -----------------------------

img_width =  int(W * ds_options.image_scale)
img_height = int(H * ds_options.image_scale)
im_channels = len(ds_options.camera_array_indices)

transforms = Compose([
	Resize(ds_options),
	Normalize(ds_options),
	SelectiveStack(ds_options),
	RandomHorizontalFlip(ds_options),
	MakeTensor(ds_options)
])

train_ds = EpiDataset(train_path, transform=transforms, options=ds_options)

x = train_ds[0]
print(x["images"].shape)