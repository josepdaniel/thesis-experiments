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
from epimodule import load_tiled_epi, load_stacked_epi
from epimodule import load_lightfield, load_relative_pose
from epimodule import DEFAULT_PATCH_INTRINSICS


class MetaData:
    """ Storage class for metadata that might be needed during evaluation"""
    def __init__(self, cameras, tgt_name, ref_names, gray, flipped):
        self.metadata = {
            "cameras": cameras,                 # List of camera indices
            "tgt_name": tgt_name,               # Filename
            "ref_names": ref_names,             # Filenames
            "gray": gray,                       # Is grayscale
            "flipped": flipped,                 # Is flipped
        }

    def getAsDict(self):
        return self.metadata


class TrainingData:
    """ Storage class for data that is needed during training.
    The training routine expects a dict with the following fields. This class ensures a consistent access API
    for different data loading modules. The __getitem__ method of any dataset class should create one of these, and
    return the dictionary obtained from TrainingData.getAsDict()
    """
    def __init__(self, tgt, tgt_formatted, ref, ref_formatted, intrinsics, pose, metadata):
        self.training_data = {
            "tgt_lf": tgt,                                      # Unprocessed grid of images
            "ref_lfs": ref,                                     # List of unprocessed grids of images
            "tgt_lf_formatted": tgt_formatted,                  # The lightfield as seen by neural nets
            "ref_lfs_formatted": ref_formatted,                 # Ref lightfields as seen by neural nets
            "pose_gt": pose,                                    # Ground truth pose between tgt and refs
            "metadata": metadata.getAsDict(),                   # Metadata (not used for training but for eval)
            "intrinsics": intrinsics,                           # Intrinsics K
            "intrinsics_inv": np.linalg.inv(intrinsics),        # Intrinsics^-1 
        }

    def getAsDict(self):
        return self.training_data


class BaseDataset(data.Dataset):
    """
    Base class for loading epi-module data-sets. Takes care of crawling the root directory and storing some common
    configuration parameters.
    """

    def __init__(self, root, cameras, gray, seed, train, sequence_length, transform, shuffle, sequence):
        np.random.seed(seed)
        random.seed(seed)

        self.samples = None
        self.cameras = cameras
        self.gray = gray
        self.root = Path(root)
        self.shuffle = shuffle
        self.transform = transform

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
            intrinsics = DEFAULT_PATCH_INTRINSICS
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

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class FocalstackLoader(BaseDataset):

    def __init__(self, root, cameras, fs_num_cameras, fs_num_planes, gray=False, seed=None, train=True,
                 sequence_length=3, transform=None, shuffle=True, sequence=None):

        super(FocalstackLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                               transform, shuffle, sequence)

        self.numCameras = fs_num_cameras
        self.numPlanes = fs_num_planes

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        tgt_focalstack = load_multiplane_focalstack(sample['tgt'], numPlanes=self.numPlanes, 
                                                    numCameras=self.numCameras, gray=self.gray)
        
        ref_focalstacks = [
            load_multiplane_focalstack(ref_img, numPlanes=self.numPlanes, numCameras=self.numCameras, gray=self.gray)
            for ref_img in sample['ref_imgs']
        ]

        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]
            tgt_focalstack, _ = self.transform(tgt_focalstack, np.zeros((3, 3)))
            ref_focalstacks = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_focalstacks]

        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]
        tgt_focalstack = torch.cat(tgt_focalstack, 0)
        ref_focalstacks = [torch.cat(ref, 0) for ref in ref_focalstacks]

        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)
        trainingdata = TrainingData(tgt_lf, tgt_focalstack, ref_lfs, ref_focalstacks, intrinsics, pose, metadata)

        return trainingdata.getAsDict()


class StackedLFLoader(BaseDataset):

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
                 transform=None, shuffle=True, sequence=None):

        super(StackedLFLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                              transform, shuffle, sequence)

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

        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)
        trainingdata = TrainingData(tgt_lf, tgt_lf, ref_lfs, ref_lfs, intrinsics, pose, metadata)

        return trainingdata.getAsDict()


class TiledEPILoader(BaseDataset):

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
                 transform=None, shuffle=True, sequence=None):
        
        super(TiledEPILoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                             transform, shuffle, sequence)

    def __getitem__(self, index):
        sample = self.samples[index]

        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        tgt_epi = load_tiled_epi(sample['tgt'])
        ref_epis = [load_tiled_epi(ref_img) for ref_img in sample['ref_imgs']]

        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])
        intrinsics = np.copy(sample['intrinsics'])

        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]
            tgt_epi, _ = self.transform(tgt_epi, np.zeros((3,3)))
            ref_epis = [self.transform(ref, np.zeros((3,3)))[0] for ref in ref_epis]

        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]
        tgt_epi = torch.cat(tuple(tgt_epi), 0)
        ref_epis = [torch.cat(ref, 0) for ref in ref_epis]

        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)
        trainingdata = TrainingData(tgt_lf, tgt_epi, ref_lfs, ref_epis, intrinsics, pose, metadata)

        return trainingdata.getAsDict()


    

def getEpiLoaders(args, train_transform, valid_transform, shuffle=True):
    train_set = TiledEPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        transform=train_transform,
        shuffle=shuffle,
    )

    val_set = TiledEPILoader(
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

def getValidationEpiLoader(args, sequence=None, transform=None, shuffle=False):
    return TiledEPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=transform,
        shuffle=shuffle,
        sequence=sequence
    )


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

