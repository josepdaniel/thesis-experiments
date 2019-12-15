import os 
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import dateutil.parser as dateparser
from . import utils as epi_utils
from .transforms import get_relative_6dof
import cv2


class EpiDatasetOptions():
    def __init__(self):

        self.debug = True
        self.grayscale = False
        self.image_scale = 1
        self.center_crop_margin = 1
        self.camera_array_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        self.with_pose = True
        self.with_intrinsics = False
        self.translation_scale = 1
        self.rotation_scale = 1
        self.step = 1
        self.backwards = True
        self.sequence_length = 7

        if type(self.step) is int:
            self.step = [self.step]

        self.sequences_with_correction = {
            "loriroo1-withpose": np.array([
                [1,  0, 0],
                [0,  0, 1],
                [0, -1, 0],
            ]),
            "fruit1-withpose": np.array([
                [1,  0, 0],
                [0,  0, 1],
                [0, -1, 0],
            ]),
            "santa1-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "santa2-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "christmastree1-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "flamingo1-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "lorikeet1-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "ornaments-mostly-straight": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang1": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang2": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang3": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang4": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang5": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang6": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang7": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
            "thegang8": np.array([
                [-1,   0, 0],
                [ 0,  -1, 0],
                [ 0,   0, 1],
            ]),
        }

        self.sequence_means = {
            "loriroo1-withpose":    23.63,
            "birb1-withpose":       42.48,
            "christmas1-withpose":  20.11,
            "christmas2-withpose":  93.34,
            "fruit1-withpose":      22.84,
            "default":              31.00,
        }

        self.sequence_stds = {
            "loriroo1-withpose":    20.13,
            "birb1-withpose":       27.86,
            "christmas1-withpose":  23.37,
            "christmas2-withpose":  56.29,
            "fruit1-withpose":      15.55,
            "default":              34.00,
        }


class EpiDataset(Dataset):

    def __init__(
        self, datadir, 
        options=EpiDatasetOptions(),
        sequences=None, 
        transform=None,
        ):

        self.datadir = datadir 
        self.sequences = sequences 
        self.transform = transform 
        self.options = options

        if self.options.with_intrinsics:
            self.intrinsics = epi_utils.get_intrinsics()
        else:
            self.intrinsics = None

        self.image_sequences = self.crawl_datadir(sequences, datadir)

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        data = {} 
        images = self.image_sequences[idx]   
        data["imagenames"] = images
        
        sequence_poses = []
        sequence_images = []

        if self.options.debug:
            print("[DEBUG]: Images are")
            for i in images:
                print("- " + str(i))

        if self.intrinsics is not None:
            data["k"] = self.intrinsics

        for i, image in enumerate(images):
            sequence_name = os.path.dirname(image).split("/")[-1]
            data["sequence_name"] = sequence_name

            if self.options.with_pose and i >= 1:
                posefile1 = images[i-1].split(".epirect")[0] + ".txt"
                posefile2 = image.split(".epirect")[0] + ".txt"
                p1 = load_pose(posefile1)
                p2 = load_pose(posefile2)
            
                t1, r1 = p1[:3], p1[3:]
                t2, r2 = p2[:3], p2[3:]
            
                # The camera wasn't always mounted the same way on the arm, correction factor for relative pose
                # is applied here
                if sequence_name in self.options.sequences_with_correction.keys():
                    relative_pose = get_relative_6dof(
                        t1, r1, t2, r2, 
                        rotation_mode='axang', 
                        correction=self.options.sequences_with_correction[sequence_name]
                    )
                    relative_pose = np.hstack([relative_pose[3:], relative_pose[:3]])
                    relative_pose *= 1000
                    sequence_poses.append(relative_pose)
                    
                    if self.options.debug:
                        print("[DEBUG] Applying correction factor")
                else:
                    relative_pose = get_relative_6dof(
                        t1, r1, t2, r2, 
                        rotation_mode='axang', 
                    )
                    relative_pose = np.hstack([relative_pose[3:], relative_pose[:3]])
                    relative_pose *= 1000
                    sequence_poses.append(relative_pose) 

            lf = epi_utils.read_rectified(image)
            sequence_images.append(lf)

        data["images"] = np.array(sequence_images)
        
        if self.options.with_pose:
            data["poses"] = np.array(sequence_poses)

        if self.transform:
            data = self.transform(data)

        return data

    def crawl_datadir(self, sequences, datadir):
        
        if sequences is None:
            sequences = os.listdir(datadir)
        elif type(sequences) == str:
            f = open(sequences, 'r')
            sequences = f.readlines()
            f.close()

        sequences = [os.path.join(datadir, s.rstrip()) for s in sequences]
        short_image_sequences = []

        for seq in sequences:
            lfs = os.listdir(seq)
            lfs = [lf for lf in lfs if lf.split(".")[-1] == "epirect"]

            # MacOS adds unicode characters to filenames apparently...
            lfs_no_unicode = [lf.encode('ascii', 'ignore').decode('utf-8') for lf in lfs]             
            timestamps = [dateparser.parse(lf.split(".")[0]) for lf in lfs_no_unicode]    
            
            sort_order = sorted(range(len(timestamps)), key=timestamps.__getitem__)            
            zipped_pairs = zip(lfs, sort_order)
            
            lfs = [lfs[i] for i in sort_order]
            lfs = [os.path.join(seq, lf) for lf in lfs]
            
            for i in range(len(lfs) - (self.options.sequence_length-1)):
                current_sequence = lfs[i:i+self.options.sequence_length]
                
                short_image_sequences.append(current_sequence)
                if self.options.backwards:
                    short_image_sequences.append(current_sequence[::-1])  # Reverse order

        return short_image_sequences

class Normalize:
    def __init__(self, options):
        self.options = options

    def __call__(self, datapoint):
        images = datapoint["images"]
        output_images = []
        for i, lf in enumerate(images):
            output_images.append(self.convertToFloat(lf, zero_centered=True))

        datapoint["images"] = np.array(output_images)
        return datapoint

    def convertToFloat(self, lf, zero_centered=False):
        lf = np.float32(lf)
        lf = lf/255

        if zero_centered:
            lf = lf - 0.5

        return lf


class Resize:
    def __init__(self, options):
        self.options = options
    
    def __call__(self, datapoint):
        images = datapoint["images"]
        output_images = []
        for i, lf in enumerate(images):
            N, H, W, C = lf.shape 
            out_w, out_h = int(W*self.options.image_scale), int(H*self.options.image_scale)
            
            outlf = np.empty([N, out_h, out_w, C]).astype(np.uint8)
            margin = self.options.center_crop_margin

            # Rescale each RGB individually
            for n in range(0, N):
                img = lf[n, margin:-margin, margin:-margin, :]
                sampled = cv2.resize(img, (out_w, out_h))
                outlf[n, :, :, :] = sampled
            output_images.append(outlf)
        
        datapoint["images"] = np.array(output_images)
        return datapoint

class SelectiveStack:
    def __init__(self, options):
        self.options = options

    def __call__(self, datapoint):
        output_images = []
        images = datapoint["images"]
        
        for lf in images:
            if len(lf.shape) == 4:
                N, H, W, C = lf.shape
            elif len(lf.shape) == 3:
                lf = np.expand_dims(lf, 3)
                N, H, W, C = lf.shape
            else:
                raise ValueError("Input must be a NxHxWxC light field")

            num_channels = C * len(self.options.camera_array_indices)
            outlf = np.empty([H, W, num_channels]).astype(np.float32)
            
            for j, n in enumerate(self.options.camera_array_indices):
                outlf[:, : ,j*C:j*C+C] = lf[n, :, :, :]
            
            output_images.append(outlf)
        datapoint["images"] = np.array(output_images)
        return datapoint

class MakeTensor:
    def __init__(self, options):
        self.options = options


    def __call__(self, datapoint):
        images = datapoint["images"]
        output_images = []

        for lf in images:
            output_images.append(
                lf.transpose([2, 0, 1])
            )

        datapoint["images"] = torch.Tensor(np.array(output_images))
        return datapoint


def load_pose(pose_file):
    with open(pose_file, 'r') as p:
        txt = p.readlines()[0]
        txt = txt.replace("(", "")
        txt = txt.replace("[", "")
        txt = txt.replace(")", "")
        txt = txt.replace("]", "")
        txt = txt.split(", ")
        txt = [float(t) for t in txt]
    return txt


if __name__ == "__main__":
    import cv2
    import sys
    import matplotlib.pyplot as plt

    ds_options = EpiDatasetOptions()
    ds_options.debug = True
    ds_options.with_pose = True
    ds_options.camera_array_indices = [8]
    ds_options.step = [1,2]
    ds_options.image_scale = 0.2
    
    transforms = Compose([
        Resize(ds_options),
        Normalize(ds_options),
        SelectiveStack(ds_options),
    ])

    ds = EpiDataset("/home/joseph/Documents/epidata/straight/train", transform=transforms, options=ds_options)
    k = ds[int(sys.argv[1])]

    images = k["images"]
    print(k["poses"])

    for i in images:
        img = i[:, :, :].squeeze()
        print(img.shape)
        plt.imshow(img)
        plt.ion()
        plt.pause(1)

          