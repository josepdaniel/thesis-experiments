import numpy as np
import torch

# 10 mm spacing between cameras
CAMERA_SPACING = 0.01


def get_sub_cam_to_center_cam_translation(cameras, camera_spacing=CAMERA_SPACING):
    """Get the relative pose from the chosen subaperture to the center subaperture.

    Args:
        cameras: which camera(s) to transform. Cameras are indexed as shown -- [B]
        camera_spacing: spacing inbetween subapertures. Assumed isotropic in s and t. 

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
        T: [4,4] homogenous transform from cam to center subaperture
    
    """
   
    def one_cam(camN):
        if camN in [0, 1, 2, 3]:
            y = camera_spacing * (4-camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [13, 14, 15, 16]:
            y = camera_spacing * (12-camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = camera_spacing * (8-camN)
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
    """Get the relative pose from the center subaperture to the chosen subaperture.

    Args:
        cameras: which camera(s) to transform. Cameras are indexed as shown -- [B]
        camera_spacing: spacing inbetween subapertures. Assumed isotropic in s and t. 

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
        T: [4,4] homogenous transform from cam to center subaperture
    
    """
   
    def one_cam(camN):
        if camN in [0, 1, 2, 3]:
            y = -1*camera_spacing * (4-camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [13, 14, 15, 16]:
            y = -1*camera_spacing * (12-camN)
            T = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif camN in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = -1*camera_spacing * (8-camN)
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