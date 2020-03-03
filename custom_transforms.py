from __future__ import division
import torch
import random
import numpy as np
import math
from scipy.misc import imresize

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            if(len(im.shape)) > 2:
                im = np.transpose(im, (2, 0, 1))
            else:
                im = im.reshape([1, im.shape[0], im.shape[1]])
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics



def get_relative_6dof(p1, r1, p2, r2, rotation_mode='axang', correction=None):
    """ 
    Relative pose between two cameras. 
    
    Arguments:
        p1, p2: world coordinate of camera 1/2
        r1, r2: rotation of camera 1/2 relative to world frame
        rotation_mode: rotation convention for r1, r2. Default 'axang'
        correction: rotation matrix correction applied when camera was mounted 
                    on the arm differently from other times
    Returns:
        t: 6 degree of freedom pose from (p1, r1) to (p2, r2) 
    """

    # Convert to rotation matrix
    if rotation_mode == "axang":
        r1 = axang_to_rotm(r1, with_magnitude=True)
        r2 = axang_to_rotm(r2, with_magnitude=True)
    elif rotation_mode == "euler":
        r1 = euler_to_rotm(r1[0], r1[1], r1[2])
        r2 = euler_to_rotm(r2[0], r2[1], r2[2])
    elif rotation_mode == "rotm":
        r1 = r1 
        r2 = r2 

    if correction is not None:
        r1 = r1 @ correction
        r2 = r2 @ correction

    r1 = r1.transpose() 
    r2 = r2.transpose()

    # Ensure translations are column vectors
    p1 = np.float32(p1).reshape(3,1)
    p2 = np.float32(p2).reshape(3,1)

    # Concatenate to transformation matrices
    T1 = np.vstack([np.hstack([r1, p1]), [0,0,0,1]])
    T2 = np.vstack([np.hstack([r2, p2]), [0,0,0,1]])
    
    relative_pose = np.linalg.inv(T1) @ T2    # [4,4] transform matrix
    rotation = relative_pose[0:3, 0:3]
    rotation = rotm_to_euler(rotation)
    translation = relative_pose[0:3, 3]

    return np.hstack((translation, rotation))


def rotm_to_euler(R):
    """
    Rotation matrix to euler angles.
    DCM angles are decomposed into Z-Y-X euler rotations.
    """ 

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def axang_to_rotm(r, with_magnitude=False):
    """ 
    Axis angle representation to rotation matrix. 
        Expect 3-vector for with_magnitude=True.
        Expect 4-vector for with_magnitude=False.
    """

    if with_magnitude:
        theta = np.linalg.norm(r) + 1e-15
        r = r / theta 
        r = np.append(r, theta)

    kx, ky, kz, theta = r

    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    vtheta = 1 - math.cos(theta)

    R = np.float32([
        [kx*kx*vtheta + ctheta,     kx*ky*vtheta - kz*stheta,   kx*kz*vtheta + ky*stheta],
        [kx*ky*vtheta + kz*stheta,  ky*ky*vtheta + ctheta,      ky*kz*vtheta - kx*stheta],
        [kx*kz*vtheta - ky*stheta,  ky*kz*vtheta + kx*stheta,   kz*kz*vtheta + ctheta   ]
    ])

    return R


def euler_to_rotm(alpha, beta, gamma):
    """
    Euler angle representation to rotation matrix. Rotation is composed in Z-Y-X order.
    
    Arguments:
        Gamma: rotation about z
        Alpha: rotation about x
        Beta: rotation about y
    """

    Rx = np.float32([
        [1,  0,                0              ],
        [0,  math.cos(alpha), -math.sin(alpha)],
        [0,  math.sin(alpha),  math.cos(alpha)]
    ])

    Ry = np.float32([
        [math.cos(beta),  0,    math.sin(beta)],
        [0,               1,    0             ],
        [-math.sin(beta), 0,    math.cos(beta)]
    ])

    Rz = np.float32([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma), math.cos(gamma),  0],
        [0,               0,                1]
    ])

    R = Rz @ Ry @ Rx
    return R

