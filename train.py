import os
import numpy as np
import pdb
import pyrender
import imageio

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from run_nerf_helpers import create_nerf, cast_to_image
from nerf_utils import create_dirs, create_mano_layer, save_latent_vectors, to8b
from models.NeuralRenderer import ImplicitRenderer
from models.loss import CustomLoss
from dataset import ImagesDataset, data_prefetcher 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)


def train(args):
    
    ## create directories and save the config file
    basedir, expname, train_save_dir = create_dirs(args)
    
    ## create MANO layer
    mano_layer = create_mano_layer(args, smplx_path='./models')

    ## create dataset loader
    data_fol = args.data_fol
    dataset = ImagesDataset(args, data_fol, device, mano_layer=mano_layer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, generator=torch.Generator(device='cuda'))
    lat_vecs = dataset.lat_vecs

    ## get the data once to get some meta data
    prefetcher = data_prefetcher(dataloader)
    batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()
    
    # get the desired height and width
    H, W = int(hwf[0][0]), int(hwf[1][0])
    #estimate H,W at the res the rays are shot
    H_render_ref, W_render_ref, focal_render_ref = int(H/args.sr_factor), int(W/args.sr_factor), hwf[2][0]/args.sr_factor		#this is the hwf at lower resolution (i.e. at the res the rays are shot)


    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, optimizer, color_cal_params, other_modules = create_nerf(args, lat_vecs, sr_input_res=[H_render_ref, W_render_ref])
    pyrenderer = pyrender.OffscreenRenderer(W_render_ref, H_render_ref)

    gain, bias = color_cal_params
    global_step = start

    sr_module = other_modules['sr_module']


    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    
    init_kwargs = {'batch_size': args.batch_size, 'chunk':args.chunk, 'mano_layer': mano_layer, 'pyrenderer': pyrenderer, 'render_patches': args.render_patches,
                    'render_full_image':args.render_full_image, 'sr_module':sr_module if args.sr_factor > 1 else None, 
                    'H_render':H_render_ref, 'W_render':W_render_ref, 'H':H, 'W':W, 'sr_factor':args.sr_factor, 'pose_cond_to_sr':args.pose_cond_to_sr}
    
    renderer = ImplicitRenderer(**init_kwargs)
    loss_fn = CustomLoss(args, device)


    for i in trange(start, args.n_iterations):
        global_step += 1
        
        if batch_dict is None:
            prefetcher = data_prefetcher(dataloader)
            batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()
        
        
        # Estimate the intrinsics (at which the rays are shot) for the current batch
        focal_render = hwf[2][0]/args.sr_factor

        ## parse batch into training inputs
        o_poses = torch.Tensor(o_poses).to(device)
        
        batch_rays = batch_dict['rays'].squeeze(0)
        train_bds = batch_dict['bds'].squeeze(0)
        target_s = batch_dict['rgb'].squeeze(0)
        target_s_sr = batch_dict['rgb_sr'].squeeze(0) if 'rgb_sr' in batch_dict else None
        acc_mask = batch_dict.get('acc_mask', None)
        acc_mask_sr = batch_dict.get('acc_mask_sr', None)
        mm_latent = batch_dict['mm_latent'].squeeze(0) if 'mm_latent' in batch_dict else None 
        
        ## get the GT MANO parameters
        mano_output = batch_dict['mano_output']
        for k in mano_output:
            mano_output[k] = mano_output[k].squeeze(0)
        
        cam_index = int(img_names[0].split('.')[0])
        t_bias, t_gain = bias[cam_index], gain[cam_index]
        
        target = {'rgb': target_s, 'rgb_sr': target_s_sr, 'acc': acc_mask}


        output = renderer(batch_rays=batch_rays, bds=train_bds, acc_mask=acc_mask, acc_mask_sr=acc_mask_sr, mm_latent=mm_latent,
                        mano_output=mano_output, focal_render=focal_render, t_gain=t_gain, t_bias=t_bias, **render_kwargs_train)

        optimizer.zero_grad()
        
        
        loss, loss_log = loss_fn(output, target)


        loss.backward()
        optimizer.step()
        
        ## for debugging
        # plot_grad_flow(render_kwargs_train['network_fn'].named_parameters(), f'vis/{expname}_gradients.png')

        
        
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        optimizer.param_groups[0]['lr'] = new_lrate
        
        if args.color_cal_lrate>0:
            new_color_lrate = args.color_cal_lrate * (decay_rate ** (global_step / decay_steps))
            optimizer.param_groups[1]['lr'] = new_color_lrate       #NOTE: the order of the param_groups is hard-coded here
        

        ################################
        


        if i%args.i_weights==args.i_weights-1:
            if lat_vecs is not None:
                save_latent_vectors(os.path.join(basedir, expname), lat_vecs, i)
            
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    # 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gain': gain, 
                    'bias': bias
                    }
            if 'network_fine' in render_kwargs_train:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            if 'texture_encoder' in render_kwargs_train:
                save_dict['texture_encoder_state_dict'] = render_kwargs_train['texture_encoder'].state_dict()
            if sr_module is not None:
                save_dict['sr_module_state_dict'] = sr_module.state_dict()
            if 'deform_net' in render_kwargs_train:
                save_dict['deform_net_state_dict'] = render_kwargs_train['deform_net'].state_dict()
            torch.save(save_dict, path)

            print('\nSaved checkpoints at', path)


        if i%args.i_img==0:
            #save the rgb and rgb_sr (if exists) images
            im = to8b(output['rgb'].detach().cpu().numpy())
            image_name = os.path.join(train_save_dir, f'{i:06}_{scan_id[0]}_cam{cam_index}.png')
            imageio.imwrite(image_name, im)
            
            if output['rgb_sr'] is not None:
                im = to8b(output['rgb_sr'].detach().cpu().numpy())
                image_name = os.path.join(train_save_dir, f'{i:06}_{scan_id[0]}_cam{cam_index}_sr.png')
                imageio.imwrite(image_name, im)
        
        
        batch_dict, hwf, o_poses, scan_id, img_names = prefetcher.next()


        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i}   Loss: {loss.item()}")  
                
            for key, value in loss_log.items():
                writer.add_scalar(key, value, i)
            
            if args.render_full_image and args.acc_loss:
                writer.add_image("real_opacity",  to8b(cast_to_image(acc_mask[...,None])), i)
                writer.add_image("rendered_opacity",  to8b(cast_to_image(output['acc'][...,None])), i)
        
        if i%5000==0:
            #log the network weights to tensorboard
            for name, param in render_kwargs_train['network_fn'].named_parameters():
                writer.add_histogram(name, param.data, i)
            if 'network_fine' in render_kwargs_train:
                for name, param in render_kwargs_test['network_fine'].named_parameters(): 
                    writer.add_histogram(name, param.data, i)
            if 'deform_net' in render_kwargs_train:
                for name, param in render_kwargs_train['deform_net'].named_parameters():
                    writer.add_histogram(name, param.data, i)
            if 'texture_encoder' in render_kwargs_train:
                for name, param in render_kwargs_train['texture_encoder'].named_parameters():
                    writer.add_histogram(name, param.data, i)
            if sr_module is not None:
                for name, param in sr_module.named_parameters():
                    writer.add_histogram(name, param.data, i)


