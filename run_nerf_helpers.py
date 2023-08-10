import numpy as np
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NeRF import NeRF
from models.superresolution import SuperresolutionHybrid
from input_encoder import read_mano_uv_obj, save_obj_for_debugging, get_uvd

try:
    from torchsearchsorted import searchsorted          #needed for heirarchical sampling
except ImportError:
    print('torchsearchsorted not found')
    pass



DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_ch=3):
    if i == -1:
        return nn.Identity(), input_ch
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_ch,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def get_rays2(H, W, focal, c2w):
    H, W = int(H), int(W)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# Ray helpers
def get_rays(H, W, focal, c2w, i ,j):
    #i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    #i = i.t()
    #j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d



# Hierarchical sampling
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    img = tensor.detach().cpu().numpy()
    return img



def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    
    def ret(inputs, additional_inputs):
        return torch.cat([fn(inputs[i:i+chunk], additional_inputs) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, fn, netchunk=1024*64, additional_inputs={}):
    """Applies network 'fn'.
    """
    outputs = batchify(fn, netchunk)(inputs, additional_inputs=additional_inputs)

    return outputs



def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    
    sigmas = raw[...,-1]  # [N_rays, N_samples, 1]
    rgbs = torch.sigmoid(raw[...,:-1])  # [N_rays, N_samples, 3/63]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigmas.shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(sigmas.shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # dists = torch.cat([dists, torch.Tensor([0.000625]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]     torch.mean(dists[:,:-1]) = 0.0008
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    alpha = 1. - torch.exp(-dists * (F.relu(sigmas + noise)))    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[...,None] * rgbs, -2)  # [N_rays, 3/63]
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)                    #[N_rays]
    disp_map = acc_map / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                mm_latent,
                additional_inputs,
                network_fn,
                network_query_fn,
                N_samples,
                points_encoding = 'xyz',
                embed_fn = None,
                deform_input = None,
                deform_output = None,
                deform_net = None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                mesh_faces=None,
                mesh_face_uv=None,):
    """Volumetric rendering.
    Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
        in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
        N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])                 # [N_rays * N_samples, 3]
    
    
    if deform_input == 'xyz':
        pts_embedded = embed_fn(pts_flat)
        hand_pose = additional_inputs["mano_output"]["hand_pose"].expand(pts_embedded.shape[0], -1)
        deformation_input = torch.cat([pts_embedded, hand_pose], -1)
        
        deformation = deform_net(deformation_input)
        
        pts_flat = pts_flat + deformation
        
        #? is the uv sampling even differentiable for this deformation model to be trained?
        #TODO add a deformation field for the fine network too
    
    hand_pose = additional_inputs["mano_output"]["hand_pose"]
    
    
    if points_encoding == 'uvd':
        mesh_vertices = additional_inputs["mano_output"]["vertices"].squeeze().contiguous()
        
        pts_uv, pts_d, intermediates = get_uvd(pts_flat, mesh_vertices, mesh_faces, mesh_face_uv)
        
        if deform_input == 'uvd':
            uvd_embedded = embed_fn(torch.cat([pts_uv, pts_d], -1))
            hand_pose = hand_pose.expand(uvd_embedded.shape[0], -1)
            deformation_input = torch.cat([uvd_embedded, hand_pose], -1)
            deformation = deform_net(deformation_input)
            
            if deform_output == 'uvd':
                pts_uv = pts_uv + deformation[:, :2]
                pts_d = pts_d + deformation[:, 2:]
                
            elif deform_output == 'xyz':
                pts_flat = pts_flat + deformation
                pts_uv, pts_d, intermediates = get_uvd(pts_flat, mesh_vertices, mesh_faces, mesh_face_uv)
    
    # save_obj_for_debugging(xyz=pts_flat.cpu().numpy(), r=np.zeros(pts_flat.shape[0]), g=np.zeros(pts_flat.shape[0]), b=np.ones(pts_flat.shape[0]), filename='vis/sampling_pts.obj')
    # save_obj_for_debugging(xyz=mesh_vertices.cpu().numpy(), r=np.ones(mesh_vertices.shape[0]), g=np.zeros(mesh_vertices.shape[0]), b=np.zeros(mesh_vertices.shape[0]), filename='vis/mano_vertices.obj')
    # pdb.set_trace()
    
    
    if points_encoding == 'xyz':
        inputs_flat = pts_flat
    elif points_encoding == 'uvd':
        inputs_flat = torch.cat([pts_uv, pts_d], dim=-1)


    #get additional inputs based on the experiment settings
    if viewdirs is not None:                                                        #[N_rays, 3]
        input_dirs = viewdirs[:,None,:].repeat(1,N_samples,1)                       #[N_rays, N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])     #[N_rays * N_samples, 3]
        inputs_flat = torch.cat([inputs_flat, input_dirs_flat], -1)

    if mm_latent is not None:                                                             #[N_rays, 8]
        mm_latent_flat = mm_latent.expand(inputs_flat.shape[0], -1)
        inputs_flat = torch.cat([inputs_flat, mm_latent_flat], -1)


    if inputs_flat.shape[0] > 0:
        raw_flat = network_query_fn(inputs_flat, network_fn, additional_inputs)          #calls run_network()
    else:
        raw_flat = torch.zeros((0, 33), device=pts_flat.device)
    

    raw = torch.reshape(raw_flat, list(pts.shape[:-1]) + [raw_flat.shape[-1]])              #[N_rays, N_samples, 4]
    
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, weights_0 = rgb_map, disp_map, acc_map, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])                 # [N_rays * (N_samples + N_importance), 3]
        
        
        #get modified inputs based on the experiment settings
        if points_encoding == 'uvd':
            pts_uv, pts_d, intermediates = get_uvd(pts_flat, mesh_vertices, mesh_faces, mesh_face_uv)
        
        if points_encoding == 'xyz':
            inputs_flat = pts_flat
        elif points_encoding == 'uvd':
            inputs_flat = torch.cat([pts_uv, pts_d], dim=-1)


        #get additional inputs based on the experiment settings
        if viewdirs is not None:                                                        #[N_rays, 3]
            input_dirs = viewdirs[:,None,:].repeat(1,N_samples+N_importance,1)                       #[N_rays, N_samples + N_imp, 3]
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])     #[N_rays * (N_samples+N_imp), 3]
            inputs_flat = torch.cat([inputs_flat, input_dirs_flat], -1)

        if mm_latent is not None:                                                             #[N_rays, 8]
            mm_latent_flat = mm_latent.expand(inputs_flat.shape[0], -1)
            inputs_flat = torch.cat([inputs_flat, mm_latent_flat], -1)
        
        run_fn = network_fn if network_fine is None else network_fine
        raw_flat = network_query_fn(inputs_flat, run_fn, additional_inputs)          #calls run_network()
        raw = torch.reshape(raw_flat, list(pts.shape[:-1]) + [raw_flat.shape[-1]])              #[N_rays, N_samples+N_imp, 4]
    
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if DEBUG:
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(rays_flat, mm_latent, additional_inputs, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], mm_latent, additional_inputs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def prune_bg_rays(rays):
    #if any of the elements in the last dim (bds) is nan, remove that row. also get the pruned indices
    pruning_mask = torch.isnan(rays).any(dim=-1)
    selection_indices = torch.nonzero(~pruning_mask).squeeze()
    rays = rays[~pruning_mask]
    return rays, selection_indices

def unprune_bg_rays(rets, n_rays, non_pruned_ind, white_bkgd=False):
    for k in rets:              #possible keys: dict_keys(['rgb_map', 'disp_map', 'acc_map', 'depth_map', 'weights', 'raw'])
        if k in ['rgb_map', 'rgb0'] and white_bkgd:
            output = torch.ones((n_rays, *rets[k].shape[1:]), device=device)
        else:
            output = torch.zeros((n_rays, *rets[k].shape[1:]), device=device)
        output[non_pruned_ind] = rets[k].type(torch.float32)        #NOTE: done because some of the outputs are 32bits and some 16bits (when using amp training). Not sure if typecasting causes an issue with amp performance gain.
        rets[k] = output
    return rets


def render(H, W, focal, chunk=1024*32, rays=None, mm_latent=None, mano_output=None,
            c2w=None, ndc=True, use_viewdirs=False, c2w_staticcam=None, bds=None, **kwargs):
    """Render rays
    Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        focal: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
        rays: array of shape [2, (batch_size/H,W), 3]. Ray origin and direction for
        each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
        camera while using other c2w argument for viewing directions.
    Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
    """

    additional_inputs = {}
    additional_inputs['mano_output'] = mano_output
    

    bd_near, bd_far = bds[...,0].unsqueeze(-1), bds[...,1].unsqueeze(-1)
    rays_o, rays_d = rays           #rays_o.shape: torch.Size([N_rand, 3])

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]    #[N_rand,3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    bd_near = torch.reshape(bd_near, [-1,1]).float()        #torch.Size([N_rand or H*W, 1])
    bd_far = torch.reshape(bd_far, [-1,1]).float()
    rays = torch.cat([rays_o, rays_d, bd_near, bd_far], -1)
    
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    

    #prune background rays (BG ray pixels have bds set to NaN)
    n_rays = rays.shape[0]
    rays, non_pruned_ind = prune_bg_rays(rays)


    #pass the rays to the NeRF module in batches
    all_ret = batchify_rays(rays, mm_latent, additional_inputs, chunk, **kwargs)
    
    #unprune the output
    all_ret = unprune_bg_rays(all_ret, n_rays, non_pruned_ind, white_bkgd=kwargs['white_bkgd'])


    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    return ret_list + [ret_dict]



def create_nerf(args, lat_vecs, sr_input_res=None):
    """Instantiate NeRF's MLP models.
    """
    
    grad_vars = []
    
    use_sr_module = args.sr_factor > 1

    nerf_input_ch = 3

    #instantiate the embedder
    embed_fn, embedded_input_ch = get_embedder(args.multires, args.i_embed, nerf_input_ch)
    #NeRF viewdirs
    embedded_input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, embedded_input_ch_views = get_embedder(args.multires_views, args.i_embed, 3)
    
    #output dimensions of the MLP
    local_nerf_output_ch = args.local_nerf_output_ch

    #latent vectors
    latent_emb_size = args.latent_size if args.use_lat_vecs else 0

    skips = [4] if args.netdepth > 4 else []
    model = NeRF(D=args.netdepth, W=args.netwidth, local_nerf_output_ch=local_nerf_output_ch,
                        embedded_input_ch=embedded_input_ch, skips=skips,
                        embedded_input_ch_views=embedded_input_ch_views, use_viewdirs=args.use_viewdirs, 
                        latent_emb_size=latent_emb_size, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
                        hand_pose_conditioning=args.hand_pose_conditioning).to(device)
    grad_vars = grad_vars + list(model.parameters())


    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, local_nerf_output_ch=local_nerf_output_ch,
                        embedded_input_ch=embedded_input_ch, skips=skips,
                        embedded_input_ch_views=embedded_input_ch_views, use_viewdirs=args.use_viewdirs, 
                        latent_emb_size=latent_emb_size, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
                        hand_pose_conditioning=args.hand_pose_conditioning).to(device)
        grad_vars = grad_vars + list(model_fine.parameters())

    if args.deform_input:
        #define an MLP to deform the sampling points
        mano_handpose_size = 45
        deform_input_size = embedded_input_ch + mano_handpose_size
        deform_net = nn.Sequential(nn.Linear(deform_input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)).to(device)
        
        # #taken from Neural Actor - should help with stable training by keeping the deformation values zero at the beginning
        deform_net[-1].weight.data *= 0
        deform_net[-1].bias.data *= 0
        grad_vars = grad_vars + list(deform_net.parameters())
    
    
    sr_module = None
    if use_sr_module:
        channels = args.sr_input_ch 
        sr_num_fp16_res = 4         #Number of fp16 layers in superresolution
        sr_kwargs = {
            "channel_base": 32768,          #most likely, this doesnt matter. the class never uses this value
            "channel_max": 512,             #most likely, this doesnt matter. the class never uses this value
            "fused_modconv_default": "inference_only"       ## Speed up training by using regular convolutions instead of grouped convolutions (from EG3D train.py)
        }
        sr_module = SuperresolutionHybrid(channels=channels, input_resolution=np.array(sr_input_res), sr_factor=args.sr_factor,
                                                  sr_num_fp16_res=sr_num_fp16_res, sr_antialias=True, **sr_kwargs)
        grad_vars = grad_vars + list(sr_module.parameters())
    
    
    if lat_vecs is not None:
        grad_vars = grad_vars + list(lat_vecs.parameters())


    network_query_fn = lambda inputs, network_fn, additional_inputs : run_network(inputs, network_fn, netchunk=args.netchunk, additional_inputs=additional_inputs)



    gain = torch.ones(args.n_cam, 3, requires_grad=(args.color_cal_lrate>0))
    bias = torch.zeros(args.n_cam, 3, requires_grad=(args.color_cal_lrate>0))
    #keep the gain and bias of the first cam identity; rest learn wrt it
    gradient_mask = torch.ones(args.n_cam, 3)
    gradient_mask[0] = 0.0
    gain.register_hook(lambda grad: grad.mul_(gradient_mask))           #this hook will be called every time a gradient with respect to the Tensor is computed
    bias.register_hook(lambda grad: grad.mul_(gradient_mask))
    color_cal_params = [gain, bias] 
    

    # Create optimizer
    if args.color_cal_lrate>0:
        optimizer = torch.optim.Adam([{'params': grad_vars, 'lr': args.lrate, 'betas': (0.9, 0.999)}, {'params': color_cal_params, 'lr': args.color_cal_lrate}])
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if sr_module is not None:
            sr_module.load_state_dict(ckpt['sr_module_state_dict'])         #when the previous model was trained in supervised mode, this will fail
        if args.deform_input:
            deform_net.load_state_dict(ckpt['deform_net_state_dict'])
        
        if ckpt['gain'] is not None:
            color_cal_params[0] = ckpt['gain']
        if ckpt['bias'] is not None:
            color_cal_params[1] = ckpt['bias']


    vt, ft, f = read_mano_uv_obj('./models/mano/MANO_UV_right.obj')
    #vt: uv coordinates of the vertices of the MANO mesh                #(891, 2), range: [0, 1]
    #ft: MANO mesh face indices for vt                                  #(1538, 3), range: [0, 890]
    #f: MANO mesh face indices for the vertices of the MANO mesh        #(1538, 3), range: [0, 777]
    mesh_faces = torch.tensor(f)                                    #NOTE: this is same as mano_layer[hand_type].faces
    mesh_face_uv = torch.tensor(vt[ft], dtype=torch.float32)        #torch.Size([1538, 3, 2])   #Neural Actor encoder.py line 1244



    ########################## 
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'points_encoding' : args.points_encoding,
        'embed_fn' : embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'deform_input' : args.deform_input,
        'deform_output' : args.deform_output,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'mesh_faces' : mesh_faces,
        'mesh_face_uv' : mesh_face_uv,
    }
    if model_fine is not None:
        render_kwargs_train['network_fine'] = model_fine
    if args.deform_input:
        render_kwargs_train['deform_net'] = deform_net

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    
    other_modules = {
        'sr_module': sr_module,
    }

    return render_kwargs_train, render_kwargs_test, start, optimizer, color_cal_params, other_modules
