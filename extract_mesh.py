import torch
import os
import numpy as np
from run_nerf_helpers import render_path


## NOTE: this code is not ready, but this is the skeleton that can be used

extract_mesh(args, render_kwargs_test, gt_cameras=gt_cameras, pyrenderer=pyrenderer,
        mano_layer=mano_layer, all_mano_params=all_mano_params,
        p_fns=p_fns, basedir=basedir, expname=expname, meta_data=meta_data, lat_vecs=lat_vecs, device=device, start=start, sr_factor=args.sr_factor, sr_module=sr_module)


#4. extract mesh out of NeRF volume with marching cube
def extract_mesh(args, render_kwargs_test, gt_cameras, pyrenderer,
                mano_layer, root_poses, hand_poses, shape_params, root_translations, hand_types,
                p_fns, basedir, expname, meta_data, lat_vecs, device, start, sr_factor, sr_module):

    print("Extracting mesh using marching cubes")            
    import mcubes

    meshsavedir = os.path.join(basedir, expname, 'mesh')
    os.makedirs(meshsavedir, exist_ok=True)
    
    N = 64
    render_kwargs_test['N_samples'] = N
    render_kwargs_test['retraw'] = True
    
    # N = 256
    # # t = torch.linspace(-100., 100., N + 1)
    # # query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
    # x = torch.linspace(-4., 4., N + 1)
    # y = torch.linspace(-4., 4., N + 1)
    # z = torch.linspace(-14., -6., N + 1)
    # query_pts = torch.stack(torch.meshgrid(x, y, z), -1)
    # sh = query_pts.shape
    # # print("query_pts: ", query_pts.shape)
    # flat = query_pts.reshape([-1, 3])
    # # maxIndex = flat.shape[0] - (flat.shape[0]%args.chunk)
    # # flat = flat[None, :maxIndex, :]
    # flat = flat[None, ...]
    # # print("flat: ", flat.shape)

    # pose_name = 'frame23308'        #unseen validation pose for iden1
    pose_name = 'frame27832'        #unseen validation pose for iden2
    pose_norm_params = {
                "root_pose": root_poses[p_fns.index(pose_name)],
                "hand_pose": hand_poses[p_fns.index(pose_name)], 
                "shape_param": shape_params[p_fns.index(pose_name)],
                "root_translation": root_translations[p_fns.index(pose_name)], 
                "hand_type": hand_types[p_fns.index(pose_name)]
                }
    
    cam_index = 12
    camera_extrinsics = gt_cameras["camera_extrinsics"][cam_index].unsqueeze(0)
    hwf = gt_cameras["hwf"][cam_index].unsqueeze(0)
    hwf_render = hwf / sr_factor
    render_bds = gt_cameras["render_bds"][pose_name][cam_index]           #list of tuples (one for each camera)
    #increase the render bds to avoid clipping
    # render_bds = (render_bds[0]*0.95, render_bds[1]*1.1)
    
    t_mm_latent = None
    if lat_vecs is not None:
        t_mm_latent = getInterpLatent(p_fns, meta_data, lat_vecs, scan1=pose_name, scan2=pose_name, ratio=1.0)
    # print("t_mm_latent.shape: ", t_mm_latent.shape)

    render_kwargs_test['inference_threshold'] = 0.02
    
    with torch.no_grad():
        rgbs, rest = render_path(camera_extrinsics, hwf_render, args.chunk, render_kwargs_test, mm_latent=t_mm_latent,
                            mano_layer=mano_layer, pose_norm_params=pose_norm_params, render_bds=render_bds, render_factor=args.render_factor, 
                            sr_module=sr_module, sr_factor=args.sr_factor, pose_cond_to_sr=args.pose_cond_to_sr)

        # pdb.set_trace()
        sigma = rest['raws'][..., -1].squeeze(0)
        # print("raw.shape: ", raw.shape)
        sigma = np.maximum(sigma, 0.)
        # print(sigma)
        # plt.hist(np.maximum(0, sigma.ravel()), log=True)
        # plt.savefig(os.path.join(meshsavedir, 'sigma_hist.png'))

        threshold = 50
        # threshold = 20
        print('fraction occupied', np.mean(sigma > threshold))
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        print('done', vertices.shape, triangles.shape)

        print(f"vertices bounds: ({np.min(vertices[:,0])}:{np.max(vertices[:,0])}), ({np.min(vertices[:,1])}:{np.max(vertices[:,1])}), ({np.min(vertices[:,2])}:{np.max(vertices[:,2])})")
        print(f"trianles bounds: ({np.min(triangles[:,0])}:{np.max(triangles[:,0])}), ({np.min(triangles[:,1])}:{np.max(triangles[:,1])}), ({np.min(triangles[:,2])}:{np.max(triangles[:,2])})")

        # x = x.cpu().detach().numpy()
        # y = y.cpu().detach().numpy()
        # z = z.cpu().detach().numpy()
        # # print(np.min(x), np.max(x))
        # # print(x)
        # vertices[:,0] = ((vertices[:,0]/N) * (np.max(x)-np.min(x))) + np.min(x)
        # vertices[:,1] = ((vertices[:,1]/N) * (np.max(y)-np.min(y))) + np.min(y)
        # vertices[:,2] = ((vertices[:,2]/N) * (np.max(z)-np.min(z))) + np.min(z)

        vertices = vertices / N
        print(f"vertices bounds: ({np.min(vertices[:,0])}:{np.max(vertices[:,0])}), ({np.min(vertices[:,1])}:{np.max(vertices[:,1])}), ({np.min(vertices[:,2])}:{np.max(vertices[:,2])})")

        mcubes.export_obj(vertices, triangles, os.path.join(meshsavedir, f"mesh_{pose_name}_{start}.obj"))

    return