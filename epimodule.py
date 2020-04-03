import math
import numpy as np
import torch
import cv2
from imageio import imread
import os
from custom_transforms import get_relative_6dof


CAMERA_SPACING = 0.04
DEFAULT_PATCH_SIZE = (160, 224)
DEFAULT_PATCH_INTRINSICS = np.array([
    [197.68828,     0,              DEFAULT_PATCH_SIZE[1]/2],
    [0,             197.68828,      DEFAULT_PATCH_SIZE[0]/2],
    [0,             0,              1]
]).astype(np.float32)


def load_as_float(path, gray, patch_size=None):
    im = imread(path).astype(np.float32)

    assert (type(patch_size) in [int, list, tuple]) or (patch_size is None)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    elif type(patch_size) in [list, tuple]:
        assert len(patch_size) == 2

    if patch_size:
        h, w = im.shape[0:2]
        x_min = math.floor(w / 2 - patch_size[1] / 2)
        y_min = math.floor(h / 2 - patch_size[0] / 2)
        im = im[y_min:y_min+patch_size[0], x_min:x_min+patch_size[1], :]

    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im


def load_lightfield(path, cameras, gray, patch_size=DEFAULT_PATCH_SIZE):
    imgs = []
    for cam in cameras:
        img_path = path.replace('/8/', '/{}/'.format(cam))
        imgs.append(load_as_float(img_path, gray, patch_size=patch_size))

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


def get_sub_cam_to_center_cam_translation(cameras, camera_spacing=CAMERA_SPACING):
    """Get the relative pose from the chosen sub-aperture to the center sub-aperture.

    Args:
        cameras: which camera(s) to transform. Cameras are indexed as shown -- [B]
        camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.

                        0   
                        1
                        2
                        3
            4  5  6  7  8  9  10 11 12 
                        13               z
                        14              /
                        15             /_____ x   
                        16             | 
                                       | 
                                       y
   
        this shows the view from BEHIND the camera, i.e. optical axis going into the page.

    Returns:
        T: [4,4] homogeneous transform from cam to center sub-aperture
    
    """

    def one_cam(camN):
        if camN in [0, 1, 2, 3]:
            y = camera_spacing * (4 - camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [13, 14, 15, 16]:
            y = camera_spacing * (12 - camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = camera_spacing * (8 - camN)
            T = np.array([
                [1, 0, 0, x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            raise ValueError("Camera must be in range [0, 16]")
        return T

    Ts = []
    for cam in cameras:
        Ts.append(one_cam(cam))

    return torch.Tensor(Ts)


def get_center_cam_to_sub_cam_translation(cameras, camera_spacing=CAMERA_SPACING):
    # T = np.linalg.inv(get_sub_cam_to_center_cam_translation(cam, camera_spacing))
    # return torch.Tensor(T)
    """Get the relative pose from the center sub-aperture to the chosen sub-aperture.

    Args:
        cameras: which camera(s) to transform. Cameras are indexed as shown -- [B]
        camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.

                        0   
                        1
                        2
                        3
            4  5  6  7  8  9  10 11 12 
                        13               z
                        14              /
                        15             /_____ x   
                        16             | 
                                       | 
                                       y
   
        this shows the view from BEHIND the camera, i.e. optical axis going into the page.

    Returns:
        T: [4,4] homogeneous transform from cam to center sub-aperture
    
    """

    def one_cam(camN):
        if camN in [0, 1, 2, 3]:
            y = -1 * camera_spacing * (4 - camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [13, 14, 15, 16]:
            y = -1 * camera_spacing * (12 - camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = -1 * camera_spacing * (8 - camN)
            T = np.array([
                [1, 0, 0, x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            raise ValueError("Camera must be in range [0, 16]")
        return T

    Ts = []
    for cam in cameras:
        Ts.append(one_cam(cam))

    return torch.Tensor(Ts)


def shift_sum(lf, shift, dof, gray):
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
        focalstack += np.sum(lf[left + right + top + bottom], 0)
    else:
        for i in left:
            img = lf[i]
            s = (8 - i) * shift
            focalstack[:, :-s:, :] += img[:, s:, :]
        for i in right:
            img = lf[i]
            s = (i - 8) * shift
            focalstack[:, s:, :] += img[:, :-s, :]
        for i in top:
            img = lf[i]
            s = (4 - i) * shift
            focalstack[:-s, :, :] += img[s:, :, :]
        for i in bottom:
            img = lf[i]
            s = (i - 12) * shift
            focalstack[s:, :, :] += img[:-s, :, :]

    focalstack = focalstack / dof
    if gray:
        focalstack = cv2.cvtColor(focalstack, cv2.COLOR_RGB2GRAY)
    return focalstack.astype(np.uint8)


def load_multiplane_focalstack(path, numPlanes, numCameras, gray):
    assert numCameras in [5, 9, 13, 17]
    assert numPlanes in [9, 7, 5, 3]

    planes = None
    if numPlanes == 9:
        planes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif numPlanes == 7:
        planes = [0, 1, 2, 3, 4, 6, 8]
    elif numPlanes == 5:
        planes = [0, 2, 4, 6, 8]
    elif numPlanes == 3:
        planes = [0, 4, 8]

    stacks = []
    lf = load_lightfield(path, gray=False, cameras=list(range(0, 17)))
    for p in planes:
        stacks.append(shift_sum(lf, p, numCameras, gray=gray))

    return stacks


def load_horizontal_epi_polar_plane_image(path, patch_size=None):
    horizontal_cameras = [5, 6, 7, 8, 9, 10, 11, 12]
    lf = load_lightfield(path, gray=True, cameras=horizontal_cameras, patch_size=patch_size)
    lf = np.array(lf)
    h, w = lf.shape[1:3]

    lf = lf.transpose([0, 2, 1])
    return lf


def load_vertical_epi_polar_plane_image(path, patch_size=None):
    vertical_cameras = [1, 2, 3, 8, 13, 14, 15, 16]
    lf = load_lightfield(path, gray=True, cameras=vertical_cameras, patch_size=patch_size)
    lf = np.array(lf)
    h, w = lf.shape[1:3]

    return lf


def load_tiled_epi(path, patch_size):
    """ Loads a tiled epipolar plane image (2D) """
    vertical = load_vertical_epi_polar_plane_image(path, patch_size).transpose(2, 0, 1)
    vertical = vertical.reshape(8*vertical.shape[0], vertical.shape[2], 1).transpose()
    return vertical


def load_stacked_epi(path, patch_size):
    """ Loads an epipolar volume. Only works if the patch is a square. """
    assert isinstance(patch_size, int)
    horizontal = load_horizontal_epi_polar_plane_image(path, patch_size=patch_size)
    vertical = load_vertical_epi_polar_plane_image(path, patch_size=patch_size)
    epi = np.concatenate([horizontal, vertical], 0)
    return epi


# Demo of how to use shiftsum
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lf1 = load_multiplane_focalstack(
        "/home/joseph/Documents/thesis/epidata/module-1-1/module1-1-png/seq2/8/0000000030.png", numPlanes=9,
        numCameras=9, gray=False)

    plt.imshow(lf1[8])
    plt.show()


