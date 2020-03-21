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


class MetaData:
    def __init__(self):
        self.metadata = {}
    def set(self, key, value):
        self.metadata[key] = value
    def get(self, key):
        return self.metadata[key]
    def getAsDict(self):
        return self.metadata


class FocalstackLoader(data.Dataset):
    """
    Arguments:
        root:                   directory where data is
        cameras:                list of cameras to build lf with
        fs_num_cameras:         num of cameras to build focal stack with
        fs_num_planes:          how many planes to focus on (one img per plane, stacked on colour channel)
        gray:                   gray
        seed:                   random seed
        train:                  is training dataset
        sequence_length:        sequence length (min 3)
        transform:              deterministic transforms (e.g. normalize, resize, toTensor)
        shuffle:                shuffle dataset
        sequence:               select one sequence, else all detected sequences are used
        random_horizontal_flip: flip all returned lfs horizontally at random
        intrinsics_file:        location of intrinsics file, default = './intrinsics.txt'

    Returns:
        tgt_lf:                 used by billinear interpolator to warp camera-wise
        ref_lfs:                used by billinear interpolator to warp camera-wise
        tgt_focalstack:         seen by depth and pose networks
        ref_focalstacks:        seen by depth and pose networks
        intrinsics:             used by photometric warper
        intrinsics_inv:         used by photometric warper
        pose:                   centerCam to centerCam pose, used for evaluation
    """

    def __init__(self, root, cameras, fs_num_cameras, fs_num_planes, gray=False, seed=None, train=True, sequence_length=3,
        transform=None, shuffle=True, sequence=None, random_horizontal_flip=False, intrinsics_file="./intrinsics.txt",
    ):

        np.random.seed(seed)
        random.seed(seed)

        self.cameras = cameras
        self.numcameras = fs_num_cameras 
        self.numplanes = fs_num_planes
        self.gray=gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.transform = transform
        self.intrinsics_file = intrinsics_file
        
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        
        if sequence is not None:
            self.scenes = [self.root/sequence/"8"]
        else:
            self.scenes = [self.root/folder[:].rstrip()/"8" for folder in open(scene_list_path)]
        self.crawl_folders(sequence_length)

        
    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(self.intrinsics_file).astype(np.float32).reshape((3, 3))
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
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]
        
        tgt_focalstack = load_multiplane_focalstack(sample['tgt'], numplanes=self.numplanes, numcameras=self.numcameras, gray=self.gray)
        ref_focalstacks = [load_multiplane_focalstack(ref_img, numplanes=self.numplanes, numcameras=self.numcameras, gray=self.gray) for ref_img in sample['ref_imgs']]
        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        
        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3,3)))
            ref_lfs = [self.transform(ref, np.zeros((3,3)))[0] for ref in ref_lfs]
            tgt_focalstack, _ = self.transform(tgt_focalstack, np.zeros((3,3)))
            ref_focalstacks = [self.transform(ref, np.zeros((33)))[0] for ref in ref_focalstacks]
        
        
        tgt_lf = torch.cat(tuple(tgt_lf), 0)       
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]    
        tgt_focalstack = torch.cat(tgt_focalstack, 0)       
        ref_focalstacks = [torch.cat(ref, 0) for ref in ref_focalstacks]     
        
        metadata = MetaData()
        metadata.set('cameras', self.cameras)
        metadata.set('tgt_name', sample['tgt'])
        metadata.set('ref_names', sample['ref_imgs'])
        metadata.set('gray', self.gray)
        metadata.set('flipped', False)

        trainingdata = {}
        trainingdata['tgt_lf'] = tgt_lf
        trainingdata['tgt_lf_formatted'] = tgt_focalstack
        trainingdata['ref_lfs'] = ref_lfs
        trainingdata['ref_lfs_formatted'] = ref_focalstacks
        trainingdata['intrinsics'] = intrinsics
        trainingdata['intrinsics_inv'] = np.linalg.inv(intrinsics)
        trainingdata['pose_gt'] = pose
        trainingdata['metadata'] = metadata.getAsDict()

        return trainingdata

    def __len__(self):
        return len(self.samples)





class StackedLFLoader(data.Dataset):
    """
    Arguments:
        root:                   directory where data is
        cameras:                list of cameras to build lf with
        gray:                   gray
        seed:                   random seed
        train:                  is training dataset
        sequence_length:        sequence length (min 3)
        transform:              deterministic transforms (e.g. normalize, resize, toTensor)
        shuffle:                shuffle dataset
        sequence:               select one sequence, else all detected sequences are used
        random_horizontal_flip: flip all returned lfs horizontally at random
        intrinsics_file:        location of intrinsics file, default = './intrinsics.txt'

    Returns:
        tgt_lf:                 used by billinear interpolator to warp camera-wise
        ref_lfs:                used by billinear interpolator to warp camera-wise
        intrinsics:             used by photometric warper
        intrinsics_inv:         used by photometric warper
        pose:                   centerCam to centerCam pose, used for evaluation
    """

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
        transform=None, shuffle=True, sequence=None, random_horizontal_flip=False, intrinsics_file="./intrinsics.txt",
    ):

        np.random.seed(seed)
        random.seed(seed)

        self.cameras = cameras
        self.gray=gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.transform = transform
        self.intrinsics_file = intrinsics_file
        
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        
        if sequence is not None:
            self.scenes = [self.root/sequence/"8"]
        else:
            self.scenes = [self.root/folder[:].rstrip()/"8" for folder in open(scene_list_path)]
        self.crawl_folders(sequence_length)

        
    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(self.intrinsics_file).astype(np.float32).reshape((3, 3))
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
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]
        
        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        
        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3,3)))
            ref_lfs = [self.transform(ref, np.zeros((3,3)))[0] for ref in ref_lfs]
        
        
        tgt_lf = torch.cat(tuple(tgt_lf), 0)       
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]     
       
        return tgt_lf, tgt_lf, ref_lfs, ref_lfs, intrinsics, np.linalg.inv(intrinsics), pose

    def __len__(self):
        return len(self.samples)
    
def getFocalstackLoaders(args, train_transform, valid_transform, shuffle=True):
    train_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=shuffle,
        sequence="seq3"
    )

    val_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=shuffle,
    )

    return train_set, val_set


def getStackedLFLoaders(args, train_transform, valid_transform, shuffle=True):
    train_set = StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        transform=train_transform, 
        shuffle=shuffle, 
    )

    val_set = StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=valid_transform, 
        shuffle=shuffle
    )

    return train_set, val_set