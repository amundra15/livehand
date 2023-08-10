import numpy as np
import pdb
import torch
import pyrender
import time
from termcolor import colored

from config_parser import *
import smplx_extended
from run_nerf_helpers import render, create_nerf
from models.superresolution import featuremap_to_rgb
from camera_utils import pose_spherical


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


#parse config
parser = config_parser()
args = parser.parse_args()

## create MANO layer
smplx_path = '../models'
mano_layer = {
            'right': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=True, num_pca_comps=45, is_Euler=False, flat_hand_mean=False),
            'left': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=False, num_pca_comps=45, is_Euler=False, flat_hand_mean=False)
                }
if 'cuda' in device.type:
    mano_layer['right'] = mano_layer['right'].to(device)
    mano_layer['left'] = mano_layer['left'].to(device)

sr_factor = args.sr_factor
hwf = np.array([512.0000,  334.0000, 1262.6090])
H_render, W_render, focal_render = int(hwf[0]/sr_factor), int(hwf[1]/sr_factor), hwf[2]/sr_factor		#this is the hwf at lower resolution (i.e. at the res the rays are shot)
hwf_render = [H_render, W_render, focal_render]
    
pyrenderer = pyrender.OffscreenRenderer(W_render, H_render)

#create models
render_kwargs_train, render_kwargs_test, _, _, color_cal_params, other_modules = create_nerf(args, None, [], sr_input_res=[H_render, W_render])
sr_module = other_modules['sr_module']


#create camera trajectory
render_poses = torch.stack([pose_spherical(theta=0, phi=-90, z=angle, radius=1.0, translate=1.0) for angle in np.linspace(-180,180,140)], 0) 	#for InterHand2.6M data

# pdb.set_trace()


pose_norm_params = {'root_pose': torch.tensor([[0.9276, -1.3251,  1.1898]], device=device),
                    'hand_pose': torch.tensor([[ 0.0233, -0.1639, -0.4504, -0.1698, -0.0679, -0.6508,  0.1125, -0.2453,
                                        -0.1829,  0.0142, -0.2303, -0.3153,  0.0748,  0.0207, -0.5773, -0.0569,
                                        -0.0740, -0.2852,  0.2955,  0.2953, -0.4380,  0.2822, -0.3277, -0.3145,
                                        0.2213, -0.2413, -0.1069, -0.1239,  0.0317,  0.6627,  0.0141,  0.0459,
                                        -0.3083, -0.0079, -0.1238, -0.1399, -0.0867,  0.0382, -0.2785, -0.3653,
                                        0.2700,  0.0090, -0.1406,  0.2361, -0.4564]], device=device), 
                    'shape_param': torch.tensor([[-2.1664,  0.0399, -0.8873,  0.0178,  0.0287, -0.0613,  0.0854,  0.1335, -0.1056, -0.0302]], device=device), 
                    'root_translation': torch.tensor([[-0.1409,  0.0211,  1.1175]], device=device), 
                    'hand_type': 'right'}



with torch.no_grad():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i, render_pose in enumerate(render_poses):
        output, disp, acc, rest = render(H_render, W_render, focal_render, chunk=args.chunk, pose_norm_params=pose_norm_params, 
                                        c2w=render_pose[:3,:4], bds=None, **render_kwargs_test)
        
        if sr_module is not None:
            rgb_lr, rgb = featuremap_to_rgb(output, sr_module, device, conditioning=None, gt_H=H_render*sr_factor, gt_W=W_render*sr_factor)
        else:
            rgb = output[...,:3]
    
        # #save the image
        # rgb = to8b(rgb.cpu().numpy())
        # rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        # cv.imwrite(f'~/Downloads/temp_sr/pose_spherical_{i:03d}.png', rgb)

    end.record()
    torch.cuda.synchronize()
    rendering_time_per_image = start.elapsed_time(end)/render_poses.shape[0]
    print(colored(f"Rendering time per image: {rendering_time_per_image:.3f} ms, {1000/rendering_time_per_image:.3f} FPS", 'yellow'))
