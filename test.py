import os
import numpy as np
import imageio
import pdb
import pyrender
from subprocess import check_output
from tqdm import trange
import torch
from torch.utils.data import DataLoader

from run_nerf_helpers import create_nerf
from nerf_utils import create_mano_layer, to8b
from models.NeuralRenderer import ImplicitRenderer
from models.loss import CustomLoss, calculate_fid_scores
from dataset import ImagesDataset, data_prefetcher 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)


def test(args, test_type):
    
    basedir = args.basedir
    expname = args.expname
    
    ## create MANO layer
    mano_layer = create_mano_layer(args, smplx_path='./models')

    ## create dataset loader
    data_fol = args.val_data_fol
    dataset = ImagesDataset(args, data_fol, device, mano_layer=mano_layer, test_type=test_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, generator=torch.Generator(device='cuda'))
    lat_vecs = dataset.lat_vecs

    ## get the data once to get some meta data
    prefetcher = data_prefetcher(dataloader)
    batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()
    
    # get the desired height and width
    H, W = int(hwf[0][0]), int(hwf[1][0])
    #estimate H,W at the res the rays are shot
    H_render_ref, W_render_ref = int(H/args.sr_factor), int(W/args.sr_factor)		#this is the hwf at lower resolution (i.e. at the res the rays are shot)


    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, optimizer, color_cal_params, other_modules = create_nerf(args, lat_vecs, sr_input_res=[H_render_ref, W_render_ref])
    pyrenderer = pyrender.OffscreenRenderer(W_render_ref, H_render_ref)

    gain, bias = color_cal_params
    global_step = start

    sr_module = other_modules['sr_module']

    init_kwargs = {'batch_size': args.batch_size, 'chunk':args.chunk, 'mano_layer': mano_layer, 'pyrenderer': pyrenderer, 'render_patches': args.render_patches,
                    'render_full_image':args.render_full_image, 'sr_module':sr_module if args.sr_factor > 1 else None, 
                    'H_render':H_render_ref, 'W_render':W_render_ref, 'H':H, 'W':W, 'sr_factor':args.sr_factor, 'pose_cond_to_sr':args.pose_cond_to_sr}
    
    renderer = ImplicitRenderer(**init_kwargs)
    loss_fn = CustomLoss(args, device)
    
    testsavedir = os.path.join(basedir, expname, f'{test_type}{start:06d}')
    os.makedirs(testsavedir, exist_ok=True)
    
    psnr_list = []
    lpips_list = []
    fid_list = []

    for i in trange(len(dataloader)):
        global_step += 1
        
        if batch_dict is None:
            prefetcher = data_prefetcher(dataloader)
            batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()
        
        
        # Estimate the intrinsics (at which the rays are shot) for the current data
        focal_render = hwf[2][0]/args.sr_factor		#this is the hwf at lower resolution (i.e. at the res the rays are shot)

        ## parse batch into training inputs
        o_poses = torch.Tensor(o_poses).to(device)
        
        batch_rays = batch_dict['rays'].squeeze(0)
        train_bds = batch_dict['bds'].squeeze(0)
        target_s = batch_dict['rgb'].squeeze(0) if 'rgb' in batch_dict else None
        target_s_sr = batch_dict['rgb_sr'].squeeze(0) if 'rgb_sr' in batch_dict else None
        acc_mask = batch_dict.get('acc_mask', None) 
        acc_mask_sr = batch_dict.get('acc_mask_sr', None)
        mm_latent = batch_dict['mm_latent'].squeeze(0) if 'mm_latent' in batch_dict else None 
        
        ## get the GT MANO parameters
        mano_output = batch_dict['mano_output']
        for k in mano_output:
            mano_output[k] = mano_output[k][0]
        
        if test_type == 'val':
            cam_index = int(img_names[0].split('.')[0])
            t_bias, t_gain = bias[cam_index], gain[cam_index]
        else:
            t_bias, t_gain = 0.0, 1.0
        
        target = {'rgb': target_s, 'rgb_sr': target_s_sr, 'acc': acc_mask}


        with torch.no_grad():
            output = renderer(batch_rays=batch_rays, bds=train_bds, acc_mask=acc_mask, acc_mask_sr=acc_mask_sr, mm_latent=mm_latent,
                            mano_output=mano_output, focal_render=focal_render, t_gain=t_gain, t_bias=t_bias, **render_kwargs_test)

            if test_type == 'val':
                loss, loss_log = loss_fn(output, target)
                psnr_list.append(loss_log['psnr/fine_network_sr'])
                lpips_list.append(loss_log['fine_network/perceptual_loss_sr'])
        

        if test_type == 'val':
            save_path = os.path.join(testsavedir, scan_id[0])
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, img_names[0])
        else:
            save_name = os.path.join(testsavedir, f"{i}.png")
        imageio.imwrite(save_name, to8b(output['rgb_sr'].cpu().numpy()))
        
        
        ## get the next batch
        batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()



    if test_type == 'val':
        fid_list = calculate_fid_scores(data_fol, testsavedir)

        avg_psnr = np.mean(psnr_list)
        avg_lpips = np.mean(lpips_list) * 1000
        avg_fid = np.mean(fid_list)
        print(f'PSNR: {avg_psnr}, LPIPS(x1000): {avg_lpips}, FID: {avg_fid}')
        
        with open(os.path.join(testsavedir, 'scores.txt'), 'w') as f:
            f.write(expname + '\n\n')
            f.write(f'PSNR: {avg_psnr:.4f}\nLPIPS(x1000): {avg_lpips:.4f}\nFID: {avg_fid:.4f}')
        print(f"Saved scores to {os.path.join(testsavedir, 'scores.txt')}")
    
    else:
        #make a video
        wd = os.getcwd()
        os.chdir(testsavedir)
        # check_output(f'ffmpeg -r 10 -pattern_type glob -i "*.png" -vb 20M {expname}_{dataset.hand_pose_id}_spiral.mp4', shell=True)
        check_output(f'gifski --fps 10 -o {expname}_{test_type}.gif *.png', shell=True)
        os.chdir(wd)
