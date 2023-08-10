import numpy as np
import os
import glob
from termcolor import colored
import torch

import smplx_extended


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def check_assertions(args):
    
    assert args.batch_size == 1, 'batch_size should be 1'
    
    if not args.per_pixel_bds:
        assert args.sample_mask_selectively, "If per_pixel_bds is not used, we need to sample the mask selectively, else there is a strong bias to just learn black everywhere."
    
    if args.acc_loss:
        print(colored("[INFO] Right now, acc loss in enforced on the FG pixels too. But we haven't studied this extensively, so you may want to look into that.", "yellow"))

    if args.deform_input is not None:
        assert args.deform_input in ['xyz', 'uvd'], "deform_input can only be 'xyz' or 'uvd'"
        print(colored(f"[INFO] Deform MLP input:{args.deform_input}, output:{args.deform_output}", "yellow"))

    if args.deform_input == 'xyz':
        assert args.deform_output == 'xyz', "For deform_input == 'xyz', deform_output must be 'xyz' (atleast for now)"

    if args.deform_input == 'uvd':
        assert args.deform_output in ['xyz', 'uvd'], "For deform_input == 'uvd', deform_output must be 'xyz' or 'uvd'"
    
    if args.sr_factor > 1:
        assert args.render_full_image, "For super-resolution, we need to work with full images and not N random rays"

    if args.render_patches:
        assert not args.render_full_image, "Can not render patches and full image together"
        assert not args.sample_mask_selectively
        assert args.N_rand == 4096, "We render 64x64 patches (hard-coded in dataset.py)"


def create_dirs(args):
    # Create log dir
    basedir = args.basedir
    expname = args.expname
    print(colored(f'Expname is {expname}', 'white', 'on_blue', attrs=['bold']))
    trainRes = 'trainRes'
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    train_save_dir = os.path.join(basedir, expname, trainRes)
    os.makedirs(train_save_dir, exist_ok=True)

    #save the args and config file if the model is being trained
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    return basedir, expname, train_save_dir


def create_mano_layer(args, smplx_path):
    if args.dataset == 'InterHand2.6M':
        use_flat_hand_mean = False
        mano_scale = 1.0
    elif args.dataset == 'Hand3Dstudio':
        use_flat_hand_mean = True
        print(colored('WARNING: Verify if we should use flat hand mean for Hand3Dstudio', 'red'))
        mano_scale = 15.0
    mano_layer = {
                'right': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=True, num_pca_comps=45, is_Euler=False, flat_hand_mean=use_flat_hand_mean, scale=mano_scale),
                'left': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=False, num_pca_comps=45, is_Euler=False, flat_hand_mean=use_flat_hand_mean, scale=mano_scale)
                }
    return mano_layer
    
    
def index_mano_params(mano_params, pose_index):
    
    pose_norm_params = {
                    "root_pose": mano_params['root_poses'][pose_index],
                    "hand_pose": mano_params['hand_poses'][pose_index], 
                    "shape_param": mano_params['shape_params'][pose_index],
                    "root_translation": mano_params['root_translations'][pose_index], 
                    "hand_type": mano_params['hand_types'][pose_index]
                    }
    
    return pose_norm_params



def getSMMLatent(p_fns, meta_data, lat_vecs, scan_id):
    #get the identity latent vector based on the scan_id (i.e. pose name)
    #for cases with no scan_id (novel poses), use 0th identity
    
    if scan_id != 'dummy':
        s_id = p_fns.index(scan_id)
        iden = torch.Tensor([meta_data[s_id,0]]).type(torch.long)
    else:
        iden = torch.Tensor([0]).type(torch.long)
    
    embedding = lat_vecs(iden)

    return embedding


def getInterpLatent(p_fns, meta_data, lat_vecs, id1=None, id2=None, scan1=None, scan2=None, steps=10):

    if id1 is None and id2 is None:
        s_id1 = p_fns.index(scan1)
        s_id2 = p_fns.index(scan2)
        id1 = torch.Tensor([meta_data[s_id1,0]]).type(torch.long)
        id2 = torch.Tensor([meta_data[s_id2,0]]).type(torch.long)

    lat1 = lat_vecs(id1)
    lat2 = lat_vecs(id2)

    interp_lats = []
    for i in range(steps):
        ratio = i/(steps-1)
        lat = (1-ratio)*lat1 + ratio*lat2
        lat = lat.view(1,-1)
        interp_lats.append(lat)

    return interp_lats


def load_latent_vectors(filename, lat_vecs):
    if not os.path.isfile(filename):
        raise Exception('latent state file "{}" does not exist'.format(filename))
    
    data = torch.load(filename)
    
    if isinstance(data["latent_codes"], torch.Tensor):
        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"]['weight'].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                lat_vecs.num_embeddings, data["latent_codes"]['weight'].size()[0]
            )
        )
        
        if not lat_vecs.embedding_dim == data["latent_codes"]['weight'].size()[1]:
            raise Exception("latent code dimensionality mismatch")
        
        for i, lat_vec in enumerate(data["latent_codes"]['weight']):
            lat_vecs.weight.data[i, :] = lat_vec
        
    else:
        lat_vecs.load_state_dict(data["latent_codes"])
    return data["epoch"]


def save_latent_vectors(experiment_directory, latent_vec, epoch):
    latent_codes_dir = os.path.join(experiment_directory, 'lat_codes')
    if not os.path.exists(latent_codes_dir):
        os.makedirs(latent_codes_dir)
    all_latents = latent_vec.state_dict()
    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, str(epoch).zfill(6)+'.npy'),
    )


def get_bounds_cameras(data_fols, poses_bounds_fn, factor, sr_factor):
# this function returns the min and max of individual camera bounds

    import pdb
    near_list = []
    far_list = []
    render_bds = {}
    
    for data_fol in data_fols:
    
        all_poses = glob.glob(os.path.join(data_fol, '*', poses_bounds_fn))

        
        for pose_file in all_poses:
            poses_arr = np.load(pose_file)
            nears = poses_arr[:,-2]
            fars = poses_arr[:,-1]  
            
            #bounds for the given pose
            pose_name = pose_file.split('/')[-2]
            render_bds[pose_name] = list(zip(nears, fars))
            
            #for estimating global bounds
            near_list.append(np.min(nears))
            far_list.append(np.max(fars))  


        #get the camera poses
        #note that the camera params remain constant across differetn poses_bounds file (and only the near and far bounds change)
        #thus simply using the last poses_arr variable
        render_training_poses =  poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])    
        # Correct rotation matrix ordering and move variable dim to axis 0
        render_training_poses = np.concatenate([render_training_poses[:, 1:2, :], -render_training_poses[:, 0:1, :], render_training_poses[:, 2:, :]], 1)
        render_training_poses = np.moveaxis(render_training_poses, -1, 0).astype(np.float32)
        camera_extrinsics = render_training_poses[:,:3,:4]
        camera_intrinsics = torch.Tensor(render_training_poses[:,:3,4]).to(device)       #[h,w,f]
        a = np.array([0,0,0,1])
        a = np.expand_dims(np.expand_dims(a, axis=0), axis=0)
        # a = np.repeat(a, 15, axis=0)
        a = np.repeat(a, camera_extrinsics.shape[0], axis=0)
        camera_extrinsics = np.concatenate((camera_extrinsics, a), axis=1)
        camera_extrinsics = torch.Tensor(camera_extrinsics).to(device)

        # if factor!=0:
        #     camera_intrinsics = camera_intrinsics/factor
        downsampling_factor = int(factor/sr_factor)
        if downsampling_factor > 1:
            camera_intrinsics = camera_intrinsics/downsampling_factor
            #make H,W to ceil int - this replicates the image resizing in the dataloader
            camera_intrinsics[:,:2] = torch.ceil(camera_intrinsics[:,:2])
        
    
    #global bounds: estimate a single near and far scalar, which is the extreme bounds across all cameras and poses
    global_bds = (np.min(near_list), np.max(far_list))

    #NOTE: we return the camera extrinsics and intrinsics only from the last identity - not sure if this is a problem
    #NOTE: camera intrinsics and extrnsics remain the same across poses, but not the render_bds
    gt_cameras = {"camera_extrinsics": camera_extrinsics,       #torch.Size([139, 4, 4])
                "hwf": camera_intrinsics,                       #torch.Size([139, 3])
                "render_bds": render_bds}                       #len(render_bds) = 139

    return global_bds, gt_cameras



def plot_grad_flow(named_parameters, im_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/12'''
    
    import matplotlib
    matplotlib.use('Agg')       #doesnt try to create an interactive window
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # print("param name: ", n)
        # if(p.requires_grad) and ("bias" not in n) and ("attention_keys" not in n) and ("views_linears" not in n):
        if(p.requires_grad) and ("weights_estimator" in n) and ("bias" not in n):
        # if(p.requires_grad) and ("nerf_modules.0.pts_linears" in n):
            # print("param name: ", n)
            # print(p.grad)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
            max_grads.append(p.grad.abs().max().detach().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(im_name)
