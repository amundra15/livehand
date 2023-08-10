import os
import numpy as np
import cv2
import torch
import pyrender

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def erode_depth(depth, kernel_size=9):
    """Erodes the depth values - this means that the depth values are reduced, moving the hand towards the camera"""
    
    #set bg to a high value st erosion will exapnd the hand and not shrink it
    bg_pixels = depth == 0
    max_depth = np.max(depth)
    modified_depth = depth.copy()
    modified_depth[bg_pixels] = max_depth + 1
    
    eroded_depth = cv2.erode(modified_depth, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    
    #set bg back to 0
    new_bg_pixels = eroded_depth == max_depth + 1
    eroded_depth[new_bg_pixels] = 0
    
    return eroded_depth


def dilate_depth(depth, kernel_size=9):
    """Dilates the depth values - this means that the depth values are increased, moving the hand away from the camera"""
    
    dilated_depth = cv2.dilate(depth, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    
    return dilated_depth


def normalise_depth(depth):
    normalized_depth = depth.copy()
    min_depth = depth[depth > 0].min()
    normalized_depth[depth > 0] = (depth[depth > 0] - min_depth) / (np.max(depth) - min_depth)
    return normalized_depth*255


def render_depth(pyrenderer, mesh, extrinsics, H, W, f):
    
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
    
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh, pose=np.eye(4))

    camera = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=W/2, cy=H/2)
    if extrinsics.shape[-2] == 3:
        extrinsics = torch.cat([extrinsics, torch.Tensor([0,0,0,1]).view(1,4)], 0)
    scene.add(camera, pose=extrinsics.cpu())

    #NOTE: this rendering might be happening on the CPU but is still okay for now since it takes about 5ms
    flags = pyrender.RenderFlags.DEPTH_ONLY
    depths = pyrenderer.render(scene, flags)
    
    return depths



def get_bds_from_depths(depths, near_buffer, far_buffer):
    
    #hardcoded kernel sizes for now
    if depths.shape[0] == 512:  kernel_size = 9
    # elif depths.shape[0] in [240, 250]:  kernel_size = 3       #Hand-3D-studio
    elif depths.shape[0] in [250, 256]:  kernel_size = 5
    elif depths.shape[0] == 128:  kernel_size = 3
    else: raise ValueError('Unknown kernel size for depth shape: {}'.format(depths.shape))
    
    eroded_depths = erode_depth(depths, kernel_size)
    dilated_depths = dilate_depth(depths, kernel_size)
    
    l_bd = eroded_depths - near_buffer
    l_bd[eroded_depths==0] = float('nan')          #these will not be inferenced
    
    u_bd = dilated_depths + far_buffer
    u_bd[dilated_depths==0] = float('nan')          #these will not be inferenced
    
    t_bds = np.concatenate([l_bd[...,None], u_bd[...,None], l_bd[...,None]], -1)    #the third entry is a dummy value
    
    return t_bds