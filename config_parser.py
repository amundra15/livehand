import configargparse
from distutils.util import strtobool

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='number of random rays per gradient step')
    parser.add_argument("--batch_size", type=int, default=1, 
                        help='number of images loaded for one iteration')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--n_cam", type=int, default=15, 
                        help='learning rate')
    parser.add_argument("--color_cal_lrate", type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--n_iterations", type=int, default=200000,
                        help='number of training iterations')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')


    parser.add_argument('--test_mode', action='store_true', 
                        help='Activate test mode')
    parser.add_argument("--render_pose_interpolation", action='store_true', 
                        help='render interpolated poses from the trained NeRF')
    parser.add_argument("--render_shape_variation", action='store_true', 
                        help='render different shapes from the trained NeRF')
    parser.add_argument("--render_pose_extrapolation", action='store_true', 
                        help='render unseen poses from the trained NeRF')
    parser.add_argument("--render_spiral", action='store_true', 
                        help='render spiral video from the trained NeRF')
    parser.add_argument("--render_iden_interpolation", action='store_true', 
                        help='render results with interpolated idenities')
    parser.add_argument("--extract_mesh", action='store_true', 
                        help='generate mesh out of the trained NeRF volume')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_custom", action='store_true', 
                        help='render custom sequence of combinations')
    parser.add_argument("--render_val", action='store_true', 
                        help='render the validation set')
    parser.add_argument("--render_factor", type=int, default=1, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset", type=str, 
                        help='options: Hand3Dstudio, InterHand2.6M')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--factor", type=int, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--load_mask", action='store_true', 
                        help='read alpha channel for input images')
    parser.add_argument("--sample_mask_selectively", action='store_true', 
                        help='use alpha channel for importance sampling of foreground')
    parser.add_argument("--sample_mask_prob", type=float, default=0.1, 
                        help='fraction of background pixels in a given batch (and not the fraction of BG pixels in an image sampled')
    parser.add_argument("--acc_loss", action='store_true', 
                        help='use alpha channel for opacity loss on background region')
    parser.add_argument("--acc_loss_weight", type=float, default=0.05,  
                        help='use alpha channel for opacity loss on background region')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of saving image')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_trainset", type=int, default=50000, 
                        help='frequency of trainset saving')
    parser.add_argument("--save_acc", action='store_true', 
                        help='save accumulation map when saving rgb images')
    parser.add_argument("--save_disp", action='store_true',
                        help='save disparity map when saving rgb images')
    parser.add_argument("--vis_error", action='store_true',
                        help='visualize error map when saving rgb images')
    parser.add_argument("--recenter_poses", action='store_true', default=False,
                        help='recenter poses')

    # New stuff
    parser.add_argument("--data_fol", nargs='+', type=str, default=[],
                        help='path for all the training data')
    parser.add_argument("--val_data_fol", nargs='+', type=str, default=[],
                        help='path for all the validation data')
    parser.add_argument("--test_data_fol", nargs='+', type=str, default=[],
                        help='path for all the test data')
    parser.add_argument('--use_lat_vecs', action='store_true', 
                        help='concatenate learnable latent vectors to NeRF inputs')
    parser.add_argument("--latent_size", type=int, default=8,
                        help='latent size')

    parser.add_argument("--poses_bounds_fn", type=str, default='poses_bounds.npy', 
                        help='poses_bounds_fn')
    parser.add_argument('--per_pixel_bds', action='store_true', 
                        help='per_pixel_bds')
    parser.add_argument('--depth_n_buffer', type=float, default=0.01,
                        help='near buffer (in m) when doing depth based per-pixel-sampling')
    parser.add_argument('--depth_f_buffer', type=float, default=0.01,
                        help='far buffer (in m) when doing depth based per-pixel-sampling')

    parser.add_argument('--opacity_reg', action='store_true', 
                        help='Forces the color weights (densities) to be close to 0 or 1')
    
    
    parser.add_argument("--validation_views", nargs='+', type=str, default=[],
                        help='Camera views not used for training')
    parser.add_argument("--interpolation_poses", nargs='+', type=str,
                        help='pose indices for pose interpolation at test time')
    
    parser.add_argument('--hand_pose_conditioning', action='store_true', help='provide MANO pose to the NeRF as input')
    parser.add_argument("--local_nerf_output_ch", type=int, help='number of channels in the output of the local MLPs')
    

    #super-resolution module related
    parser.add_argument('--render_full_image', action='store_true', 
                        help='Training works on full RGB images and not N random rays')
    parser.add_argument('--render_patches', action='store_true', 
                        help='Render square patches of the image')
    parser.add_argument("--sr_factor", type=int, default=1, 
                        help='upscaling factor for the output feature map. Accepts values 2, 4')
    parser.add_argument("--sr_input_ch", type=int, default=32, help='number of channels in the input to the SR module')
    parser.add_argument("--pose_cond_to_sr", action='store_true', help="provide pose conditioning to the SR module")
    parser.add_argument('--perceptual_loss', action='store_true', 
                        help='Perceptual loss on the rendered image')
    parser.add_argument("--perceptual_loss_weight", type=float, default=1.0) 


    #encodings
    parser.add_argument("--deform_input", type=str, default=None, help='options: "xyz", "uvd"')
    parser.add_argument("--deform_output", type=str, default=None, help='options: "xyz", "uvd"')
    parser.add_argument("--points_encoding", type=str, help='options: "xyz", "uvd"')
    

    parser.add_argument("--description", type=str, help='experiment description')

    return parser
