import math

import torch.utils.data as data
import numpy as np
import random
import cv2
import torch
import os

from imageio import imread
from path import Path
from epimodule import load_multiplane_focalstack
from epimodule import load_epi_polar_plane_image
from epimodule import load_lightfield, load_relative_pose


class MetaData:
    def __init__(self):
        self.metadata = {}

    def set(self, key, value):
        self.metadata[key] = value

    def get(self, key):
        return self.metadata[key]

    def getAsDict(self):
        return self.metadata


class BaseDataset(data.Dataset):
    """
    Base class for loading epi-module data-sets. Takes care of crawling the root directory and storing some common
    configuration parameters.
    """

    def __init__(self, root, cameras, gray, seed, train, sequence_length, transform, shuffle, sequence, intrinsics):
        np.random.seed(seed)
        random.seed(seed)

        self.samples = None
        self.cameras = cameras
        self.gray = gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.transform = transform
        self.intrinsics_file = intrinsics   # TODO: Revert this

        if train:
            scene_list_path = self.root / 'train.txt'
        else:
            scene_list_path = self.root / 'val.txt'

        if sequence is not None:
            self.scenes = [self.root / sequence / '8']
        else:
            self.scenes = [self.root / folder[:].rstrip() / '8' for folder in open(scene_list_path)]

        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(self.intrinsics_file).astype(np.float32).reshape((3, 3))  # TODO: Revert this
            imgs = sorted(scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}  # TODO: Revert this
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        if self.shuffle:
            random.shuffle(sequence_set)

        self.samples = sequence_set

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FocalstackLoader(BaseDataset):

    def __init__(self, root, cameras, fs_num_cameras, fs_num_planes, gray=False, seed=None, train=True,
                 sequence_length=3, transform=None, shuffle=True, sequence=None, intrinsics_file="./intrinsics.txt"):

        super(FocalstackLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, transform, shuffle,
                                               sequence, intrinsics_file)

        self.numCameras = fs_num_cameras
        self.numPlanes = fs_num_planes

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        tgt_focalstack = load_multiplane_focalstack(sample['tgt'], numPlanes=self.numPlanes, numCameras=self.numCameras,
                                                    gray=self.gray)
        ref_focalstacks = [
            load_multiplane_focalstack(ref_img, numPlanes=self.numPlanes, numCameras=self.numCameras, gray=self.gray)
            for ref_img in sample['ref_imgs']]
        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]
            tgt_focalstack, _ = self.transform(tgt_focalstack, np.zeros((3, 3)))
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


class StackedLFLoader(BaseDataset):

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
                 transform=None, shuffle=True, sequence=None, intrinsics_file="./intrinsics.txt"):

        super(StackedLFLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, transform, shuffle,
                                              sequence, intrinsics_file)

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]

        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]

        metadata = MetaData()
        metadata.set('cameras', self.cameras)
        metadata.set('tgt_name', sample['tgt'])
        metadata.set('ref_names', sample['ref_imgs'])
        metadata.set('gray', self.gray)
        metadata.set('flipped', False)

        trainingdata = {}
        trainingdata['tgt_lf'] = tgt_lf
        trainingdata['tgt_lf_formatted'] = tgt_lf
        trainingdata['ref_lfs'] = ref_lfs
        trainingdata['ref_lfs_formatted'] = ref_lfs
        trainingdata['intrinsics'] = intrinsics
        trainingdata['intrinsics_inv'] = np.linalg.inv(intrinsics)
        trainingdata['pose_gt'] = pose
        trainingdata['metadata'] = metadata.getAsDict()

        return trainingdata

    def __len__(self):
        return len(self.samples)


class EPILoader(data.Dataset):
    """
    Arguments:
        root:                   directory where data is
        cameras:                which cameras to warp with
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
                 transform=None, shuffle=True, sequence=None, random_horizontal_flip=False,
                 intrinsics_file="./intrinsics.txt",
                 ):

        np.random.seed(seed)
        random.seed(seed)

        self.cameras = cameras
        self.gray = gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.transform = transform
        self.intrinsics_file = intrinsics_file

        scene_list_path = self.root / 'train.txt' if train else self.root / 'val.txt'

        if sequence is not None:
            self.scenes = [self.root / sequence / "8"]
        else:
            self.scenes = [self.root / folder[:].rstrip() / "8" for folder in open(scene_list_path)]
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(self.intrinsics_file).astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        if self.shuffle:
            random.shuffle(sequence_set)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, gray=self.gray, patch_size=128)
        ref_lfs = [load_lightfield(ref, self.cameras, gray=self.gray, patch_size=128) for ref in sample['ref_imgs']]
        tgt_lf_formatted = load_epi_polar_plane_image(sample['tgt'], patch_size=128)
        ref_lfs_formatted = [load_epi_polar_plane_image(ref_img, patch_size=128) for ref_img in sample['ref_imgs']]

        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        if self.transform is not None:
            tgt_lf_formatted, _ = self.transform(tgt_lf_formatted, np.zeros((3, 3)))
            ref_lfs_formatted = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs_formatted]
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]

        tgt_lf_formatted = torch.cat(tgt_lf_formatted, 0).permute(2, 0, 1)
        ref_lfs_formatted = [torch.cat(ref, 0).permute(2, 0, 1) for ref in ref_lfs_formatted]
        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]

        metadata = MetaData()
        metadata.set('cameras', self.cameras)
        metadata.set('tgt_name', sample['tgt'])
        metadata.set('ref_names', sample['ref_imgs'])
        metadata.set('gray', self.gray)
        metadata.set('flipped', False)

        trainingdata = dict()
        trainingdata['tgt_lf'] = tgt_lf
        trainingdata['tgt_lf_formatted'] = tgt_lf_formatted
        trainingdata['ref_lfs'] = ref_lfs
        trainingdata['ref_lfs_formatted'] = ref_lfs_formatted
        trainingdata['intrinsics'] = intrinsics
        trainingdata['intrinsics_inv'] = np.linalg.inv(intrinsics)
        trainingdata['pose_gt'] = pose
        trainingdata['metadata'] = metadata.getAsDict()

        return trainingdata

    def __len__(self):
        return len(self.samples)


def getEpiLoaders(args, train_transform, valid_transform, shuffle=True):
    train_set = EPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        transform=train_transform,
        shuffle=shuffle,
    )

    val_set = EPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=valid_transform,
        shuffle=shuffle
    )

    return train_set, val_set


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


def getValidationFocalstackLoader(args, sequence=None, transform=None, shuffle=False):
    val_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=shuffle,
        sequence=sequence
    )

    return val_set


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


def getValidationStackedLFLoader(args, sequence=None, transform=None, shuffle=False):
    val_set = StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=transform,
        shuffle=shuffle,
        sequence=sequence
    )
    return val_set

