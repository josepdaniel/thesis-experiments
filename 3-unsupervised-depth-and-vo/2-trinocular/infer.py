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

from lfmodels import LFDispNet as DispNetS
from lfmodels import LFPoseNet as PoseNet
from utils import tensor2array
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("--depthnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--posenet",  required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
# parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

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

    dataset = SequenceFolder(
        args.dataset_dir, 
        cameras=[7, 8, 9],
        gray=True,
        sequence_length=3, 
        shuffle=False, 
        train=False, 
        transform=transform)

    disp_net = DispNetS(in_channels=3).to(device)
    weights = torch.load(args.depthnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    pose_net = PoseNet(in_channels=3, nb_ref_imgs=2, output_exp=False).to(device)
    weights = torch.load(args.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    for i, (tgt, tgt_lf, ref, ref_lf, k, kinv) in enumerate(dataset):
        print(i, end="\r")
        tgt = tgt.unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in ref]
        tgt_lf = tgt_lf.unsqueeze(0).to(device)
        ref_lf = [r.unsqueeze(0).to(device) for r in ref_lf]
        output = disp_net(tgt_lf)
        exp, pose = pose_net(tgt_lf, ref_lf)
        
        outdir = os.path.join(args.output_dir, "{}.png".format(i))
        plt.imsave(outdir, output.cpu().numpy()[0,0,:,:])
        poses.append(pose[0,1,:].cpu().numpy())

    outdir = os.path.join(args.output_dir, "poses.npy")
    np.save(outdir, poses)


        

#     dataset_dir = Path(args.dataset_dir)
#     output_dir = Path(args.output_dir)
#     output_dir.makedirs_p()

#     if args.dataset_list is not None:
#         with open(args.dataset_list, 'r') as f:
#             test_files = [dataset_dir/file for file in f.read().splitlines()]
#     else:.
#         test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

#     print('{} files to test'.format(len(test_files)))

#     for file in tqdm(test_files):

#         img = imread(file, pilmode="RGB").astype(np.float32)

#         h,w,_ = img.shape
#         if (not args.no_resize) and (h != args.img_height or w != args.img_width):
#             img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
#         img = np.transpose(img, (2, 0, 1))

#         tensor_img = torch.from_numpy(img).unsqueeze(0)
#         tensor_img = ((tensor_img/255 - 0.5)/0.5).to(device)

#         output = disp_net(tensor_img)[0]

#         file_path, file_ext = file.relpath(args.dataset_dir).splitext()
#         file_name = '-'.join(file_path.splitall())
                                     
#         if args.output_disp:
#             disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
#             imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), disp[0,:,:])
#         if args.output_depth:
#             depth = 1/output
#             depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
#             imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), depth[0,:,:])


if __name__ == '__main__':
    main()
