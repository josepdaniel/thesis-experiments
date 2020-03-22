import os
import torch
import custom_transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataloader import getValidationFocalstackLoader, getValidationStackedLFLoader
from tqdm import tqdm
from lfmodels import LFDispNet as DispNetS
from lfmodels import LFPoseNet as PoseNet
from utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Pkl file containing training configuration")
parser.add_argument("--seq", required=True, type=str, help="Name of sequence to perform inference on")
parser.add_argument("--use-latest-not-best", action="store_true",
                    help="Use the latest set of weights rather than the best")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    poses = []
    args = parser.parse_args()
    config = load_config(args.config)

    output_dir = os.path.join(config.save_path, "results", args.seq)

    if args.use_latest_not_best:
        config.posenet = os.path.join(config.save_path, "posenet_checkpoint.pth.tar")
        config.dispnet = os.path.join(config.save_path, "dispnet_checkpoint.pth.tar")
        output_dir = output_dir + "-latest"
    else:
        config.posenet = os.path.join(config.save_path, "posenet_best.pth.tar")
        config.dispnet = os.path.join(config.save_path, "dispnet_best.pth.tar")

    os.makedirs(output_dir)

    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Preserve compatibility with old config.pkl files.
    # Old ones labelled config.focalstack as either True or False
    # New models don't have the 'focalstack' field, instead they have lfformat=[focalstack/stack]
    try:
        if config.lfformat == 'focalstack':
            dataset = getValidationFocalstackLoader(config, args.seq, transform, shuffle=False)
            print("Loading images as focalstack")
        elif config.lfformat == 'stack':
            dataset = getValidationStackedLFLoader(config, args.seq, transform, shuffle=False)
            print("Loading images as stack")

    except AttributeError:
        if ('focalstack' not in config):
            dataset = getValidationStackedLFLoader(config, args.seq, transform, shuffle=False)
            print("Loading images as stack")
        elif config.focalstack:
            dataset = getValidationFocalstackLoader(config, args.seq, transform, shuffle=False)
            print("Loading images as focalstack")
        else:
            dataset = getValidationStackedLFLoader(config, args.seq, transform, shuffle=False)
            print("Loading images as stack")

    input_channels = dataset[0][1].shape[0]

    disp_net = DispNetS(in_channels=input_channels).to(device)
    weights = torch.load(config.dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    pose_net = PoseNet(in_channels=input_channels, nb_ref_imgs=2, output_exp=False).to(device)
    weights = torch.load(config.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    for i, (tgt, tgt_lf, ref, ref_lf, k, kinv, pose_gt) in enumerate(dataset):
        print("{:03d}/{:03d}".format(i + 1, len(dataset)), end="\r")
        tgt = tgt.unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in ref]
        tgt_lf = tgt_lf.unsqueeze(0).to(device)
        ref_lf = [r.unsqueeze(0).to(device) for r in ref_lf]
        output = disp_net(tgt_lf)
        exp, pose = pose_net(tgt_lf, ref_lf)

        outdir = os.path.join(output_dir, "{:06d}.png".format(i))
        plt.imsave(outdir, output.cpu().numpy()[0, 0, :, :])
        poses.append(pose[0, 1, :].cpu().numpy())

    outdir = os.path.join(output_dir, "poses.npy")
    np.save(outdir, poses)
    print("\nok")


if __name__ == '__main__':
    main()
