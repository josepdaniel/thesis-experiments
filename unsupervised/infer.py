import os
import torch
import custom_transforms
import argparse
import matplotlib.pyplot as plt 
import numpy as np
from dataloader import SequenceFolder
from tqdm import tqdm
from lfmodels import LFDispNet as DispNetS
from lfmodels import LFPoseNet as PoseNet
from utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Pkl file containing training configuration")
parser.add_argument("--seq", required=True, type=str, help="Name of sequence to perform inference on")
parser.add_argument("--use-latest-not-best", action="store_true", help="Use the latest set of weights rather than the best")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():

    poses = []
    args = parser.parse_args()
    config = load_config(args.config)

    output_dir = os.path.join(config.save_path, "results", args.seq)
    config.posenet = os.path.join(config.save_path, "posenet_best.pth.tar")
    config.dispnet = os.path.join(config.save_path, "dispnet_best.pth.tar")

    os.makedirs(output_dir)

    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = SequenceFolder(
        config.data, 
        cameras=config.cameras,
        gray=config.gray,
        sequence_length=config.sequence_length, 
        shuffle=False, 
        train=False, 
        transform=transform,
        sequence=args.seq,
    )

    input_channels = dataset[0][1].shape[0]   

    disp_net = DispNetS(in_channels=input_channels).to(device)
    weights = torch.load(config.dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    pose_net = PoseNet(in_channels=9, nb_ref_imgs=2, output_exp=False).to(device)
    weights = torch.load(config.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    for i, (tgt, tgt_lf, ref, ref_lf, k, kinv) in enumerate(dataset):
        print("{:03d}/{:03d}".format(i+1, len(dataset)), end="\r")
        tgt = tgt.unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in ref]
        tgt_lf = tgt_lf.unsqueeze(0).to(device)
        ref_lf = [r.unsqueeze(0).to(device) for r in ref_lf]
        output = disp_net(tgt_lf)
        exp, pose = pose_net(tgt_lf, ref_lf)
        
        outdir = os.path.join(output_dir, "{:06d}.png".format(i))
        plt.imsave(outdir, output.cpu().numpy()[0,0,:,:])
        poses.append(pose[0,1,:].cpu().numpy())


    outdir = os.path.join(output_dir, "poses.npy")
    np.save(outdir, poses)
    print("\nok")

if __name__ == '__main__':
    main()
