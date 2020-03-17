import numpy as np
import cv2
import os

from imageio import imread
from path import Path
from custom_transforms import get_relative_6dof

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


def shiftsum(lf, shift, dof, gray):
        if type(lf) is list:
            lf = np.array(lf)
        assert (lf.shape[0] == 17)

        if dof == 17:
            left = [4, 5, 6, 7]
            right = [9, 10, 11, 12]
            top = [0, 1, 2, 3]
            bottom = [13, 14, 15, 16]
        elif dof == 13:
            left = [5, 6, 7]
            right = [9, 10, 11] 
            top = [1, 2, 3]
            bottom = [13, 14, 15]
        elif dof == 9:
            left = [6, 7]
            right = [9, 10] 
            top = [2, 3]
            bottom = [13, 14]
        elif dof == 5:
            left = [7]
            right = [9] 
            top = [3]
            bottom = [13]
        else:
            raise ValueError("Cannot focus at depth {}".format(dof))
        
        focalstack = lf[8].astype(np.float32)
        if shift == 0:
            focalstack += np.sum(lf[left+right+top+bottom], 0)
        else:
            for i in left:
                img = lf[i]
                s = (8-i)*shift
                focalstack[:, :-s:, :] += img[:, s:, :]
            for i in right:
                img = lf[i]
                s = (i-8)*shift
                focalstack[:, s:, :] += img[:, :-s, :]
            for i in top:
                img = lf[i]
                s = (4-i)*shift
                focalstack[:-s, :, :] += img[s:, :, :]
            for i in bottom:
                img = lf[i]
                s = (i-12)*shift
                focalstack[s:, :, :] += img[:-s, :, :]

        focalstack = focalstack/dof
        if gray:
            focalstack = cv2.cvtColor(focalstack, cv2.COLOR_RGB2GRAY)
        return focalstack.astype(np.uint8)


def load_multiplane_focalstack(path, numplanes, numcameras, gray):

    assert numcameras in [5, 9, 13, 17]
    assert numplanes in [9, 7, 5, 3]

    if numplanes == 9:
        planes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif numplanes == 7:
        planes = [0, 1, 2, 3, 4, 6, 8]
    elif numplanes == 5:
        planes = [0, 2, 4, 6, 8]
    elif numplanes == 3:
        planes = [0, 4, 8]
    
    stacks = []
    lf = load_lightfield(path, gray=False, cameras=list(range(0, 17)))
    for p in planes:
        stacks.append(shiftsum(lf, p, numcameras, gray=gray))

    return stacks

# Demo of how to use shiftsum
if __name__ == "__main__":
    lf = load_multiplane_focalstack("/home/joseph/Documents/thesis/epidata/module-1-1/module1-1-png/seq2/8/0000000030.png", numplanes=9, numcameras=17, gray=False)

    plt.imshow(lf[8])
    plt.show()