import torch
from sequence_folders import SequenceFolder
import sys
sys.path.insert(0, "../")
import custom_transforms
from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os

from models import DispNetS
from models import PoseExpNet as PoseNet
from utils import tensor2array
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("--depthnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--posenet",  required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():

    poses = []
    args = parser.parse_args()

    if not (os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir)

    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = SequenceFolder(args.dataset_dir, sequence_length=5, shuffle=False, train=False, transform=transform)

    disp_net = DispNetS().to(device)
    weights = torch.load(args.depthnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    pose_net = PoseNet(nb_ref_imgs=4, output_exp=False).to(device)
    weights = torch.load(args.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    for i, (tgt, ref, k, kinv) in enumerate(dataset):
        print(i, end="\r")
        tgt = tgt.unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in ref]
        output = disp_net(tgt)
        exp, pose = pose_net(tgt, ref)
        
        outdir = os.path.join(args.output_dir, "{:03d}.png".format(i))
        plt.imsave(outdir, output.cpu().numpy()[0,0,:,:])
        poses.append(pose[0,2,:].cpu().numpy())

    outdir = os.path.join(args.output_dir, "poses.npy")
    np.save(outdir, poses)



if __name__ == '__main__':
    main()
