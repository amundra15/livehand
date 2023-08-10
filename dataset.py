import os
import numpy as np 
import sys 
import pdb
import torch
from termcolor import colored
import pyrender
from random import randint
import pickle
import trimesh
from torch.utils.data import Dataset

from load_dataset import load_llff_data
from run_nerf_helpers import get_rays
from depth_renderer import render_depth, get_bds_from_depths
from camera_utils import pose_spherical
from nerf_utils import load_latent_vectors, getSMMLatent, getInterpLatent, index_mano_params, get_bounds_cameras

try:
    sys.path.append('torch_sampling/')
    from torch_sampling import choice
except ImportError:
    print('torch_sampling not found')
    pass



def get_random_patch_indices(rendering_pixels, patch_size, H, W):
    
    #bounding box of the rendering pixels
    r_first_row = rendering_pixels[0].min()
    r_last_row = rendering_pixels[0].max()
    r_first_col = rendering_pixels[1].min()
    r_last_col = rendering_pixels[1].max()
    
    #relace the bounding box by the patch size
    sampling_start_row = max(0, r_first_row - patch_size/2)
    sampling_end_row = min(H, r_last_row + patch_size/2)
    sampling_start_col = max(0, r_first_col - patch_size/2)
    sampling_end_col = min(W, r_last_col + patch_size/2)
    
    patch_start_row = randint(sampling_start_row, sampling_end_row - patch_size)
    patch_start_col = randint(sampling_start_col, sampling_end_col - patch_size)
    
    sel_inds = torch.stack([torch.arange(patch_start_row*W + patch_start_col + i*W, patch_start_row*W + patch_start_col + i*W + 64) for i in range(64)]).view(-1)

    return sel_inds


def get_biased_fg_indices(mask, sample_mask_prob, N_rand):
    
    fg_inds = (mask==1).nonzero(as_tuple=True)[0]
    bg_inds = (mask==0).nonzero(as_tuple=True)[0]

    fg_prob = 1.0 - sample_mask_prob

    if fg_inds.shape[0]>0 and bg_inds.shape[0]>0:
        fg_pixels = int(N_rand*fg_prob)
        bg_pixels = N_rand - fg_pixels 
        fg_sel_inds = choice(fg_inds, fg_pixels, True)
        bg_sel_inds = choice(bg_inds, bg_pixels, True)
        sel_inds = torch.cat([fg_sel_inds, bg_sel_inds], 0)
    elif fg_inds.shape[0]>0:
        bg_pixels = 0 
        fg_pixels = N_rand - bg_pixels 
        sel_inds = choice(fg_inds, fg_pixels, True)
    else:
        fg_pixels = 0 
        bg_pixels = N_rand - fg_pixels 
        sel_inds = choice(bg_inds, bg_pixels, True)
    
    return sel_inds


class ImagesDataset(Dataset):

    def __init__(self, args, data_fols, device, mano_layer=None, test_type=None):
        self.args = args 
        self.data_fols = data_fols 
        self.device = device 
        self.validation_views = args.validation_views
        self.mano_layer = mano_layer
        self.p_fls = []
        self.p_fns = []
        self.init()	
        self.ii = None
        self.jj = None 
        self.test_type = test_type
        
        if args.per_pixel_bds:
            self.pyrenderer = None          #will be initialized later

        if args.render_patches:
            self.patch_size = 64         #TODO: make this a command line argument
        
        
        if test_type not in {'spiral', 'iden', 'pose', 'shape', 'mesh', 'custom'}:
            #sample from training or validation dataset
            self.sample_from_dataset = True
            self.seq_len = len(self.all_imgs)
            
        else:
            ## render based on test time parameters
            self.sample_from_dataset = False
            
            self.global_bounds, gt_cameras = get_bounds_cameras(data_fols, args.poses_bounds_fn, args.factor, args.sr_factor)
            self.hwf_ref = [512, 334, 1270.0317]        #NOTE: hard-coded for Hand3Dstudio data for now
            self.hwf_render_ref = [int(self.hwf_ref[0]/args.sr_factor), int(self.hwf_ref[1]/args.sr_factor), self.hwf_ref[2]/args.sr_factor]		#this is the hwf at lower resolution (i.e. at the res the rays are shot)
            
        
            if test_type == 'spiral':
                #get spiral camera poses
                interpolation_steps = 81
                if self.args.dataset == 'Hand3Dstudio':
                    self.render_poses = torch.stack([pose_spherical(theta=angle, phi=0, z=100, radius=11.5, translate=-10.0) for angle in np.linspace(-180,180,interpolation_steps)], 0) 	#for Hand3Dstudio data
                elif self.args.dataset == 'InterHand2.6M': 
                    self.render_poses = torch.stack([pose_spherical(theta=angle, phi=0, z=100, radius=1.0, translate=1.0) for angle in np.linspace(-180,180,interpolation_steps)], 0) 	#for InterHand2.6M data
                else:
                    print(f"Unknown dataset: {self.args.dataset}")

                #get mano params
                hand_pose_id = 'frame23032'
                self.mano_params = index_mano_params(self.all_mano_params, self.p_fns.index(hand_pose_id))
                
                self.seq_len = len(self.render_poses)
            
            
            elif test_type == 'pose':
                #get camera pose
                cam_index = 0
                self.render_poses = gt_cameras["camera_extrinsics"][cam_index]
                
                #interpolate between mano params
                interpolation_poses = self.args.interpolation_poses if self.args.interpolation_poses else self.p_fns[0:8]
                interpolation_steps = 21
                self.mano_params = []
                for i in range(len(interpolation_poses)):
                    pose1_index = self.p_fns.index(interpolation_poses[i])
                    pose2_index = self.p_fns.index(interpolation_poses[(i+1)%len(interpolation_poses)])
                    self.mano_params.extend(self.get_interpolated_pose(pose1_index, pose2_index, interpolation_steps))
                
                self.seq_len = len(self.mano_params)
    
    
            elif test_type == 'shape':
                #get camera pose
                cam_index = 0
                self.render_poses = gt_cameras["camera_extrinsics"][cam_index]
                
                #get mano params
                hand_pose_id = 'frame23032'
                self.mano_params = index_mano_params(self.all_mano_params, self.p_fns.index(hand_pose_id))
                
                interpolation_steps = 30
                self.mano_params = self.get_shape_variation(self.mano_params, interpolation_steps)
                
                self.seq_len = len(self.mano_params)
    

            elif test_type == 'iden':
                #get camera pose
                cam_index = 12
                self.render_poses = gt_cameras["camera_extrinsics"][cam_index]
                
                #get mano params
                hand_pose_id = 'frame23164'
                self.mano_params = index_mano_params(self.all_mano_params, self.p_fns.index(hand_pose_id))
                
                id1 = torch.Tensor([0]).type(torch.long)
                id2 = torch.Tensor([1]).type(torch.long)
                self.seq_len = 30
                self.latent_vecs = getInterpLatent(self.p_fns, self.meta_data, self.lat_vecs, id1=id1, id2=id2, steps=self.seq_len)
    
    
            elif test_type == 'custom':                
                #get camera pose
                cam_index = 12
                self.render_poses = gt_cameras["camera_extrinsics"][cam_index]
                
                #interpolate between mano params
                interpolation_poses = self.args.interpolation_poses if self.args.interpolation_poses else self.p_fns[0:8]
                interpolation_steps = 21
                self.mano_params = []
                for i in range(len(interpolation_poses)):
                    pose1_index = self.p_fns.index(interpolation_poses[i])
                    pose2_index = self.p_fns.index(interpolation_poses[(i+1)%len(interpolation_poses)])
                    self.mano_params.extend(self.get_interpolated_pose(pose1_index, pose2_index, interpolation_steps))
                
                self.seq_len = len(self.mano_params)
                
                #get interpolated latent vectors
                id1 = torch.Tensor([0]).type(torch.long)
                id2 = torch.Tensor([1]).type(torch.long)
                self.latent_vecs = getInterpLatent(self.p_fns, self.meta_data, self.lat_vecs, id1=id1, id2=id2, steps=30)[15]
                
                
    
    def init(self):
        self.all_imgs = self.getAllImgs(self.data_fols)
        
        #load mano params
        self.all_mano_params = self.loadMANOparams()

        self.lat_vecs = None
        if self.args.use_lat_vecs:
            num_latentcodes, self.meta_data = self.getNumSMMLatCodes()
            #create latent vectors
            self.lat_vecs = torch.nn.Embedding(num_latentcodes, self.args.latent_size , max_norm=1.0).to(self.device)
            torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 1.0)
            
            #load saved latent codes if available
            latent_vec_ckpts = []
            # ckpt path explicitly specified
            ft_path = self.args.ft_path
            if ft_path is not None and ft_path != 'None':
                folder = os.path.dirname(ft_path)
                index = os.path.splitext(os.path.basename(ft_path))[0]
                if os.path.exists(os.path.join(folder, 'lat_codes')):
                    latent_vec_ckpts = [os.path.join(folder,'lat_codes',f"{index}.npy")]
            else:
                # search for ckpt in the experiment folder
                if os.path.isdir(os.path.join(self.args.basedir, self.args.expname, 'lat_codes')):
                    latent_vec_ckpts = [os.path.join(self.args.basedir, self.args.expname, 'lat_codes', f) for f in sorted(os.listdir(os.path.join(self.args.basedir, self.args.expname, 'lat_codes'))) if 'npy' in f]
            if len(latent_vec_ckpts) > 0 and not self.args.no_reload:
                ckpt_path = latent_vec_ckpts[-1]
                load_latent_vectors(ckpt_path, self.lat_vecs)
                print(f"Loaded latent codes from {ckpt_path}")
                # print("Loaded saved latent vectors: ", self.lat_vecs.weight)


    def __len__(self):
        return self.seq_len
    
    
    def getAllImgs(self, data_fols):
        
        out = []
        
        for data_fol in data_fols:
        
            subfolders = [f.path for f in os.scandir(data_fol) if f.is_dir()]
            print(f'Total poses in {data_fol}: {len(subfolders)}')
            
            all_imgs = sorted([os.path.join(x, 'images', f) for x in subfolders for f in os.listdir(os.path.join(x, 'images')) if f.endswith('png')])
            print(f'Total images in {data_fol}: {len(all_imgs)}')
            
            p_fls = sorted(subfolders)
            p_fns = [os.path.basename(pfl) for pfl in p_fls]
            self.p_fls.extend(p_fls)
            self.p_fns.extend(p_fns)
            
            for img in all_imgs:
                img_id = os.path.basename(img)
                img_id = img_id.split('.')[0]          #eg. '99'
                if img_id not in self.validation_views:
                    out.append(img)

        return out 
    

    def loadMANOparams(self):
        
        p_fls = self.p_fls
        device = self.device
        
        sample_articulation = os.path.join(p_fls[0], 'mesh', 'MANO_params.pkl')
        with open(sample_articulation, 'rb') as f:
            sample_hand_param = pickle.load(f)
        # print(sample_hand_param.keys())         #dict_keys(['pose', 'trans', 'shape', 'hand_type'])
        sample_rootPose = sample_hand_param['pose'][:, :3]
        sample_handPose = sample_hand_param['pose'][:, 3:]
        sample_shape = sample_hand_param['shape']
        sample_trans = sample_hand_param['trans']

        root_poses = torch.empty((len(p_fls),*(sample_rootPose.shape)), dtype=torch.float, device=device)       #torch.Size([N, 1, 3])
        hand_poses = torch.empty((len(p_fls),*(sample_handPose.shape)), dtype=torch.float, device=device)       #torch.Size([N, 1, 45])
        shape_params = torch.empty((len(p_fls),*(sample_shape.shape)), dtype=torch.float, device=device)        #torch.Size([N, 1, 10])
        root_trans = torch.empty((len(p_fls),*(sample_trans.shape)), dtype=torch.float, device=device)          #torch.Size([N, 1, 3])
        hand_types = [None] * len(p_fls)
        
        
        for index, p_fl in enumerate(p_fls):
            
            articulation_file = os.path.join(p_fl, 'mesh', 'MANO_params.pkl')
            
            with open(articulation_file, 'rb') as f:
                hand_param = pickle.load(f)

            root_pose = torch.tensor(hand_param['pose'][:, :3], dtype=torch.float, device=device)
            hand_pose = torch.tensor(hand_param['pose'][:, 3:], dtype=torch.float, device=device)
            shape_param = torch.tensor(hand_param['shape'], dtype=torch.float, device=device)
            root_tran = torch.tensor(hand_param['trans'], dtype=torch.float, device=device)
            hand_type = hand_param['hand_type']
            
            root_poses[index] = root_pose
            hand_poses[index] = hand_pose
            shape_params[index] = shape_param
            root_trans[index] = root_tran
            hand_types[index] = hand_type
        
        
        mano_params = {
                'root_poses': root_poses,
                'hand_poses': hand_poses,
                'shape_params': shape_params,
                'root_translations': root_trans,
                'hand_types': hand_types
                }
        
        return mano_params


    def get_shape_variation(self, mano_params, steps):
        
        scale_factor = 0.7
        new_shape1 = mano_params["shape_param"].clone() * (1-scale_factor)
        new_shape2 = mano_params["shape_param"].clone() * (1+scale_factor)
        
        interpolated_mano_params = []
        for i in range(steps+1):
            new_shape_param = (new_shape1*(1-i/steps)) + (new_shape2*(i/steps))
            interpolated_mano_params.append({"shape_param": new_shape_param, "hand_pose": mano_params["hand_pose"], "root_pose": mano_params["root_pose"], "root_translation": mano_params["root_translation"], "hand_type": mano_params["hand_type"]})
    
        return interpolated_mano_params
    
    
    def get_interpolated_pose(self, pose1_index, pose2_index, steps):
        
        if not torch.equal(self.all_mano_params["shape_params"][pose1_index], self.all_mano_params["shape_params"][pose2_index]):
            print("[WARNING] Shape parameters are not equal for the start and the end pose")
        
        fraction = 1/steps
        interpolated_mano_params = []
        
        for i in range(steps+1):
            interp_root_pose = (self.all_mano_params["root_poses"][pose1_index]*(1-i*fraction)) + (self.all_mano_params["root_poses"][pose2_index]*i*fraction)
            interp_hand_pose = (self.all_mano_params["hand_poses"][pose1_index]*(1-i*fraction)) + (self.all_mano_params["hand_poses"][pose2_index]*i*fraction)
            interp_root_translation = (self.all_mano_params["root_translations"][pose1_index]*(1-i*fraction)) + (self.all_mano_params["root_translations"][pose2_index]*i*fraction)

            interp_mano_params = {
                                "root_pose": interp_root_pose,
                                "hand_pose": interp_hand_pose, 
                                "shape_param": self.all_mano_params["shape_params"][pose1_index],
                                "root_translation": interp_root_translation, 
                                "hand_type": self.all_mano_params["hand_types"][pose1_index]
                                }

            interpolated_mano_params.append(interp_mano_params)
        
        return interpolated_mano_params


    def getNumSMMLatCodes(self):
        
        p_fls = self.p_fls
        
        identities = set()
        for file in p_fls:
            identity = file.split('/')[-3]
            identities.add(identity)
        identities = sorted(identities)
        print("identities: ", identities)

        mapIdExp = torch.zeros([len(p_fls), 1], dtype=torch.int64)
        for i in range(len(p_fls)):
            file = p_fls[i]
            identity = file.split('/')[-3]
            mapIdExp[i] = identities.index(identity)
        
        return len(identities), mapIdExp


    def prepare_batch(self, H, W, focal, poses, images, depths, mano_mesh, bds, mask):
        
        if self.ii is None:
            i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
            self.ii = i.t().to(self.device)
            self.jj = j.t().to(self.device)
        
        #get the rays to be shot
        rays_o,rays_d = get_rays(H, W, focal, poses[0,:3,:4], self.ii, self.jj)
        rays = torch.stack([rays_o,rays_d], 0) # [ro+rd, H, W, 3]
        #NOTE: we get the rays at the final res (i.e. after super-res). later, we sub-sample from it based on the sr_factor

        rays_rgb = torch.cat([rays, images], 0) # [3 (ro+rd+rgb), H, W, 3]
        
        
        ## Get the bounds
        if self.args.per_pixel_bds:
            if depths is None:
                #if depth file is not provided, compute coarse depth by rendering the MANO pose
                if self.pyrenderer is None:
                    self.pyrenderer = pyrender.OffscreenRenderer(W, H)

                depths = render_depth(self.pyrenderer, mano_mesh, poses[0,:3,:4], H, W, focal)      #NOTE: this is happening on CPU

            
            t_bds = get_bds_from_depths(depths, near_buffer=self.args.depth_n_buffer, far_buffer=self.args.depth_f_buffer)        #[H,W,3]
            t_bds = torch.Tensor(t_bds).to(self.device)
            t_bds = t_bds.unsqueeze(0)           #torch.Size([1, H, W, 3])
            
            rays_rgb = torch.cat([rays_rgb, t_bds], 0)
            
        else:
            #Use the per-image global bounds
            rays_bds = torch.cat([torch.Tensor(bds),torch.zeros(1,1)],1)
            rays_bds = rays_bds.expand(1,rays_rgb.shape[1],rays_rgb.shape[2],3)
            rays_rgb = torch.cat([rays_rgb, rays_bds], 0)
        
        
        #append the BG mask
        if self.args.acc_loss:
            acc_mask = mask.repeat((1, 1, 1, 3))
            rays_rgb = torch.cat([rays_rgb, acc_mask], 0) # [5 (ro+rd+rgb+bounds+mask), H, W, 3] 
        
    
        rays_rgb = rays_rgb.permute(1,2,0,3) # [H, W, info_rays_rgb, 3]
        
        
        return rays_rgb


    def getSampleData(self, samp_img):
        
        args = self.args
        images, poses, bds, distortion_mask, depths = load_llff_data(samp_img, args.dataset, args.factor, recenter=args.recenter_poses, 
                                                                    load_mask=args.load_mask, sr_factor=args.sr_factor,
                                                                    poses_bounds_fn=args.poses_bounds_fn)
        
        if images is None:
            return None, None, None
        
        if depths is not None:
            depths = torch.Tensor(depths).to(self.device)

        images = torch.Tensor(images).to(self.device)		#torch.Size([H,W,4])
        images = images/255
        images = images.view(1,images.shape[0],images.shape[1],images.shape[2])

        o_poses = poses.copy()
        poses  = torch.Tensor(poses).to(self.device)            #torch.Size([1, 3, 5])
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        
        bds[...,0] = np.maximum(bds[...,0] - 1., np.min(bds[...,0]))
        bds[...,1] = np.minimum(bds[...,1] + 1., np.max(bds[...,1]))
        
        if args.load_mask:
            mask = images[...,-1:].clone()
            if distortion_mask is not None:
                mask[0,:,:,0][~distortion_mask]=2	#these pixels will be completely ignored
        else:		
            #create a mask with all 1s
            mask = torch.empty(images.shape[0], images.shape[1], images.shape[2], 1).fill_(1.)

        if args.white_bkgd:
            images = images[...,:3] * images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3] * images[...,-1:] + (1.-images[...,-1:])*0.0

        N_rand = args.N_rand
        
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]			#the intrinsics for the final image
        #NOTE: we get the rays at the final res (i.e. after super-res). later, we sub-sample from it based on the sr_factor


        ## index the GT MANO parameters
        pose_name = os.path.basename(os.path.dirname(os.path.dirname(samp_img)))
        pose_index = self.p_fns.index(pose_name)
        mano_params = index_mano_params(self.all_mano_params, pose_index)
        
        #make a MANO forward pass
        with torch.no_grad():            
            mano_output = self.mano_layer[mano_params['hand_type']](global_orient=mano_params['root_pose'],
                                                                    hand_pose=mano_params['hand_pose'],
                                                                    betas=mano_params['shape_param'], 
                                                                    transl=mano_params['root_translation'],
                                                                    # scale=1.0,
                                                                    return_full_pose = True,
                                                                    return_as_dict=True
                                                                    )
            
            mano_mesh = mano_output['vertices'].squeeze().cpu().numpy()
            mano_mesh = trimesh.Trimesh(mano_mesh, self.mano_layer['right'].faces, process=False)


        rays_rgb = self.prepare_batch(H, W, focal, poses, images, depths, mano_mesh, bds, mask)        # [H, W, info_rays_rgb, 3]

        rays_rgb = rays_rgb.view(W*H,-1,3)		#torch.Size([W*H, info_rays_rgb, 3])
        mask = mask.view(W*H)					#torch.Size([W*H])


        if args.render_full_image:
            ## set bounds to NaN for the invalid pixels
            invalid_pixels = (mask == 2)
            rays_rgb[invalid_pixels, 3, :] = torch.tensor(float('nan'), device=self.device)
            
        else:
            #sample a subset of rays based on different sampling strategies
            
            if args.render_patches:
                valid_bds = ~ torch.isnan(t_bds[0]).any(dim=-1)     #TODO: gett_bds from indexing rays_rgb
                valid_pixels = (mask != 2).view(H,W)
                rendering_pixels = (valid_bds & valid_pixels).nonzero(as_tuple=True)
                sel_inds = get_random_patch_indices(rendering_pixels, self.patch_size, H, W)
            
                #if all selected indices are invalid, sample again
                while torch.isnan(rays_rgb[sel_inds, 3, :]).all():
                    sel_inds = get_random_patch_indices(rendering_pixels, self.patch_size, H, W)
                
                
            elif args.sample_mask_selectively and args.load_mask:
                #we sample only a fraction of BG pixels (defined with the loaded alpha channel)
                sel_inds = get_biased_fg_indices(mask, args.sample_mask_prob, N_rand)
            
            else:
                if args.per_pixel_bds:
                    #sample from pixels with valid (non-nan) bounds and valid pixels (mask != 2)
                    valid_bds = ~ torch.isnan(t_bds[0]).any(dim=-1)
                    valid_pixels = (mask != 2).view(H,W)
                    valid_indices = (valid_bds & valid_pixels).view(-1).nonzero(as_tuple=True)[0]
                else:
                    #sample from all valid image pixels (mask != 2)
                    valid_indices = (mask!=2).nonzero(as_tuple=True)[0]
                sel_inds = choice(valid_indices, N_rand, True)
            
            
            rays_rgb = rays_rgb[sel_inds]			#torch.Size([N_rand, info_rays_rgb, 3])

        batch = torch.movedim(rays_rgb, -2, 0)				#torch.Size([info_rays_rgb, N_rand, 3])


        # create a dict of batch data
        batch_dict = {}
        batch_dict['mano_output'] = mano_output
        
        if self.args.use_lat_vecs:
            scan_id = os.path.basename(os.path.dirname(os.path.dirname(samp_img)))
            batch_dict['mm_latent'] = getSMMLatent(self.p_fns, self.meta_data, self.lat_vecs, scan_id)
        
        batch_rays = batch[:2]
        if args.sr_factor > 1:
            #in case of a SR module, this will subsample the rays
            batch_rays = batch_rays.reshape(2, H, W, 3)
            batch_rays = batch_rays[:,::args.sr_factor,::args.sr_factor,:]
            batch_rays = batch_rays.reshape(2, -1, 3)
        batch_dict['rays'] = batch_rays
        
        
        rgb = batch[2]
        if args.render_full_image:
            rgb = rgb.view(H, W, -1)            #convert tensors to 2D
            
            if args.sr_factor > 1:
                batch_dict['rgb_sr'] = rgb
                #create resized versions of the tensors
                rgb = torch.moveaxis(rgb, -1, 0)
                rgb = torch.squeeze(torch.nn.functional.interpolate(rgb[None], size=(H//args.sr_factor, W//args.sr_factor),
                                                    mode='bilinear', align_corners=False, antialias=True))
                rgb = torch.moveaxis(rgb, 0, -1)

        elif args.render_patches:
            rgb = rgb.view(self.patch_size, self.patch_size, -1)
        
        batch_dict['rgb'] = rgb
        
        
        train_bds = batch[3]
        if args.sr_factor > 1:
            #in case of a SR module, this will subsample the rays
            train_bds = train_bds.reshape(H,W,3)
            train_bds = train_bds[::args.sr_factor,::args.sr_factor,:]
            train_bds = train_bds.reshape(-1,3)
        batch_dict['bds'] = train_bds
        
        
        if args.acc_loss:
            acc_mask = batch[4][...,0] > 0.5
            assert torch.all(torch.logical_or(acc_mask==0, acc_mask==1)), "acc_mask has values other than 0 and 1 (possibly 2, which is for invalid pixels)"
            #note that downsampling the seg mask leads to float values. and thresholding to 0 was giving bigger masks than original, so doinf it at 0.5
            
            if args.render_full_image:
                acc_mask = acc_mask.view(H, W)          #convert tensors to 2D
                
                if args.sr_factor > 1:
                    batch_dict['acc_mask_sr'] = acc_mask
                    #create downsampled versions of the tensors
                    acc_mask = acc_mask[::args.sr_factor,::args.sr_factor]         #torch.Size([125, 188])

            elif args.render_patches:
                acc_mask = acc_mask.view(self.patch_size, self.patch_size)
            
            batch_dict['acc_mask'] = acc_mask


        return batch_dict, hwf, o_poses


    def getInterpolatedData(self, index):
        
        if self.test_type == 'spiral': 
            c2w = self.render_poses[index][:3,:4].unsqueeze(0)
            mano_params = self.mano_params
        
        elif self.test_type in ['pose', 'shape']:
            c2w = self.render_poses[:3,:4].unsqueeze(0)
            mano_params = self.mano_params[index]
        
        elif self.test_type == 'iden':
            c2w = self.render_poses[:3,:4].unsqueeze(0)
            mano_params = self.mano_params
        
        elif self.test_type == 'custom':
            c2w = self.render_poses[:3,:4].unsqueeze(0)
            mano_params = self.mano_params[index]
        
        #make a MANO forward pass
        with torch.no_grad():
            mano_output = self.mano_layer[mano_params['hand_type']](global_orient=mano_params['root_pose'],
                                                        hand_pose=mano_params['hand_pose'],
                                                        betas=mano_params['shape_param'], 
                                                        transl=mano_params['root_translation'],
                                                        # scale=1.0,
                                                        return_full_pose = True,
                                                        return_as_dict=True
                                                        )
            
            mano_mesh = mano_output['vertices'].squeeze().cpu().numpy()
            mano_mesh = trimesh.Trimesh(mano_mesh, self.mano_layer['right'].faces, process=False)

        dummy_images = torch.zeros((1, self.hwf_render_ref[0], self.hwf_render_ref[1], 3))
        bounds = None if self.args.per_pixel_bds else self.global_bounds
        batch = self.prepare_batch(self.hwf_render_ref[0], self.hwf_render_ref[1], self.hwf_render_ref[2], c2w, dummy_images, None, mano_mesh, bounds, None)
        
        batch = batch.view(self.hwf_render_ref[0]*self.hwf_render_ref[1],-1,3)		#torch.Size([W*H, info_rays_rgb, 3])
        batch = torch.movedim(batch, -2, 0)				#torch.Size([info_rays_rgb, N_rand, 3])
        
        
        # create a dict of batch data
        batch_dict = {}
        batch_dict['mano_output'] = mano_output
        batch_dict['rays'] = batch[:2]
        batch_dict['bds'] = batch[3]
        
        if self.args.use_lat_vecs:
            if self.test_type == 'iden':
                batch_dict['mm_latent'] = self.latent_vecs[index]
            elif self.test_type == 'custom':
                batch_dict['mm_latent'] = self.latent_vecs
            else:
                batch_dict['mm_latent'] = getSMMLatent(self.p_fns, self.meta_data, self.lat_vecs, 'dummy')
        # print(f"latent vec for iter {index}: {batch_dict['mm_latent']}")

        return batch_dict, self.hwf_ref, c2w

    def __getitem__(self, index):
        
        if self.sample_from_dataset:
            samp_img = self.all_imgs[index]

            scan_id = os.path.basename(os.path.dirname(os.path.dirname(samp_img)))
            img_name = os.path.basename(samp_img)
        
            batch_dict, hwf, o_poses = self.getSampleData(samp_img)

            while batch_dict is None:
                index = randint(0,len(self.all_imgs)-1)
                samp_img = self.all_imgs[index]
                
                scan_id = os.path.basename(os.path.dirname(os.path.dirname(samp_img)))
                batch_dict, hwf, o_poses = self.getSampleData(samp_img)

        else:
            ## use arbitrary test parameters
            batch_dict, hwf, o_poses = self.getInterpolatedData(index)
            scan_id, img_name = "dummy", "dummy"


        return batch_dict, hwf, o_poses, scan_id, img_name


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_batch_dict, self.next_hwf, self.next_o_poses, self.next_scan_id, self.next_img_name = next(self.loader)
        except StopIteration:
            self.next_batch_dict = None
            self.next_hwf= None
            self.next_o_poses = None
            self.next_scan_id= None
            self.next_img_name = None
            return
        
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_dict = self.next_batch_dict
        hwf = self.next_hwf
        o_poses = self.next_o_poses
        scan_id = self.next_scan_id
        img_name = self.next_img_name
        self.preload()
        return batch_dict, hwf, o_poses, scan_id, img_name
