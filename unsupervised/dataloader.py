import torch.utils.data as data
import numpy as np
import random
import cv2
import torch

from imageio import imread
from path import Path

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


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
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
    ):

        np.random.seed(seed)
        random.seed(seed)
        self.cameras = cameras
        self.gray=gray
        self.root = Path(root)
        self.shuffle = shuffle
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

        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]
        
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

        return tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
