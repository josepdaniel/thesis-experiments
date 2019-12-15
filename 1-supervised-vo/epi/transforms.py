import math
import numpy as np 
from numpy.linalg import inv


def convert_to_homogenous(coords):
    pass


def vect6dof_to_matrix():
    pass

def apply_rotation(r, v, type='euler'):
    """ 
    Apply a rotation r to the vector v

    Arguments:
        r: rotation 
        v: vector
        type: 'euler' or 'rotm'
    
    Returns:
        v rotated by r
    """
    
    pass



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
        r1 = euler_to_rotm(r1)
        r2 = euler_to_rotm(r2)
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
    
    relative_pose = inv(T1) @ T2    # [4,4] transform matrix
    rotation = relative_pose[0:3, 0:3]
    rotation = rotm_to_euler(rotation)
    translation = relative_pose[0:3, 3]

    return np.hstack((translation, rotation))



def tool_to_camera_frame(t, r):
    """
    Convert a translation and rotation from the tools frame to the cameras frame
    """

    cvt_to_camera_axes = np.float32([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])

    t = cvt_to_camera_axes @ t
    r = cvt_to_camera_axes @ r
    return t, r
    

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


def euler_to_rotm(gamma, beta, alpha):
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