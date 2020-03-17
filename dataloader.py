import torch.utils.data as data
import numpy as np
import random
import cv2
import torch
import os

from imageio import imread
from path import Path
from custom_transforms import get_relative_6dof
from focalstack import load_multiplane_focalstack

import matplotlib.pyplot as plt

def load_as_float(path, gray):
    im = imread(path).astype(np.float32)
    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im

def load_lightfield(path, cameras, gray):
    imgs = []
    for cam in cameras:
        imgpath = path.replace('/8/', '/{}/'.format(cam))
        imgs.append(load_as_float(imgpath, gray))

    return imgs


def load_relative_pose(tgt, ref):
    # Get the number in the filename - super hacky
    sequence_name = os.path.join("/", *tgt.split("/")[:-2])
    tgt = int(tgt.split("/")[-1].split(".")[-2])
    ref = int(ref.split("/")[-1].split(".")[-2])
    pose_file = np.load(os.path.join(sequence_name, "poses_gt_absolute.npy"))
    tgt_pose = pose_file[tgt, :]
    ref_pose = pose_file[ref, :]
    rel_pose = get_relative_6dof(tgt_pose[:3], tgt_pose[3:], ref_pose[:3], ref_pose[3:], rotation_mode='euler')
    return rel_pose

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)

        Can load images as focal stack, must pass in arguments lf_format='focalstack', num_cameras, num_planes.
    """

    def __init__(self, root, 
        cameras=[8], 
        gray=False, 
        seed=None, 
        train=True, 
        sequence_length=3, 
        transform=None, 
        target_transform=None, 
        shuffle=True,
        sequence=None,
        lf_format='stack',           # Parameters to change if using focal stack only
        num_cameras=None,            # ========
        num_planes=None              # ========
    ):

        np.random.seed(seed)
        random.seed(seed)
        self.cameras = cameras
        self.gray=gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.lfformat = lf_format
        self.numcameras = num_cameras 
        self.numplanes = num_planes
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        
        if sequence is not None:
            self.scenes = [self.root/sequence/"8"]
        else:
            self.scenes = [self.root/folder[:].rstrip()/"8" for folder in open(scene_list_path)]

        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt("./intrinsics.txt").astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        if self.shuffle:
        	random.shuffle(sequence_set)
        	
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'], False)
        ref_imgs = [load_as_float(ref_img, False) for ref_img in sample['ref_imgs']]
        
        if self.lfformat == 'stack':
            tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
            ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        elif self.lfformat == 'focalstack':
            tgt_lf = load_multiplane_focalstack(sample['tgt'], numplanes=self.numplanes, numcameras=self.numcameras, gray=self.gray)
            ref_lfs = [load_multiplane_focalstack(ref_img, numplanes=self.numplanes, numcameras=self.numcameras, gray=self.gray) for ref_img in sample['ref_imgs']]

        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))

            # Lazy reuse of existing function
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3,3)))
            ref_lfs = [self.transform(ref, np.zeros((3,3)))[0] for ref in ref_lfs]

            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        
        tgt_lf = torch.cat(tgt_lf, 0)       # Concatenate lightfield on colour channel
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]     

        return tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, np.linalg.inv(intrinsics), pose

    def __len__(self):
        return len(self.samples)
