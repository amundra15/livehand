import numpy as np
import os, imageio
import natsort 
from scipy.spatial.transform import Rotation as R


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        ext = imgs[0].split('.')[-1]
        
        if r != 1:

            print('Minifying', r, basedir)
        
            # args = ' '.join(['mogrify','-channel RGBA', '-resize', resizearg,  '-format', 'png', '*.{}'.format(ext)])       #this makes the RGB values 0 at places with full transparency
            args = f'for f in ./*.{ext}; do convert "$f" -channel RGBA -separate -resize {resizearg} -combine "${{f%.*}}.{ext}"; done'
            # print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')



def getSampInd(samp_img):
    
    basedir= os.path.dirname(samp_img)
    if not os.path.isdir(basedir):
        return None
    fls = [os.path.join(basedir, f) for f in natsort.natsorted(os.listdir(basedir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    return fls.index(samp_img)


def _load_data(samp_img, factor=1, load_imgs=True, load_mask=False, poses_bounds_fn="poses_bounds.npy"):

    basedir = os.path.dirname(os.path.dirname(samp_img))
    samp_ind_in_scan = getSampInd(samp_img)

    if samp_ind_in_scan is None:
        return None, None, None, None

    poses_arr = np.load(os.path.join(basedir, poses_bounds_fn))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])		#first 4 cols are cam2world  rotation and translation; last is focal length width height
    bds = poses_arr[:, -2:].transpose([1,0])
    
    sfx = ''
    if factor > 1:
        sfx = f'_{factor}'
        _minify(basedir, factors=[factor])
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    # natsort sorts images according to the image idx, ['10.png', '2.png', '1.png'] --> ['1.png', '2.png', '10.png']
    imgfiles = [os.path.join(imgdir, f) for f in natsort.natsorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        print(f"for folder {imgdir}")
        return
    
    
    def imread(f):
        if f.endswith('png'):
            sh = imageio.imread(f).shape
            if sh[2] == 4:
                return imageio.imread(f, pilmode='RGBA', ignoregamma=True)
            else:
                return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    if load_mask:
        imgs = imgs = imread(imgfiles[samp_ind_in_scan])[...,:4]#/255.
        #convert alpha channel to 0 or 255
        imgs[...,3] = (imgs[...,3] > 128).astype(np.uint8) * 255
    else:
        imgs = imgs = imread(imgfiles[samp_ind_in_scan])[...,:3]#/255.]

    depths = None
    # if self.args.per_pixel_bds:
    #     # depth_fol = os.path.join( os.path.dirname(os.path.dirname(imgfiles[samp_ind_in_scan])), 'depths')
    #     # t_filename = str(samp_ind_in_scan) + '.npz'
    #     # depth_file = os.path.join(depth_fol, t_filename)
    #     # depths = np.load(depth_file)
    #     # depths = depths['arr_0']
    

    distortion_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(imgfiles[samp_ind_in_scan]))), 'valid_pixel_masks.npy')
    distortion_mask = None
    if os.path.isfile(distortion_file):
        distortion_masks = np.load(distortion_file, allow_pickle=True)
        distortion_mask = distortion_masks.item().get(str(samp_ind_in_scan))
        assert distortion_mask.shape[0] == imgs.shape[0] and distortion_mask.shape[1] == imgs.shape[1]
        # try:
        #     assert distortion_mask.shape[0] == imgs.shape[0] and distortion_mask.shape[1] == imgs.shape[1]
        # except:
        #     pdb.set_trace()
    # else:
    #     print('No distortion mask found for image {}'.format(imgfiles[samp_ind_in_scan]))

    sh = imgs.shape
    if len(sh)<3:
        return None, None, None, None
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        raise ValueError('load_imgs=False not supported')
        return poses, bds

    arr_ind     = np.array([samp_ind_in_scan]).astype(np.int64) #np.arange(poses.shape[2])
    #s_poses = np.take(poses, arr_ind, 2)
    s_bds   = np.take(bds, arr_ind, 1)
    
    return poses, s_bds, imgs, arr_ind, distortion_mask, depths


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def getInterpolatedPose(pose1, pose2, ratio):
    out = pose1*0

    rotMat1 = R.from_matrix(pose1[:,0:3])  
    rotMat2 = R.from_matrix(pose2[:,0:3])  
    angles1 = R.as_rotvec(rotMat1) 
    angles2 = R.as_rotvec(rotMat2) 
    angle   = (angles1*ratio) + (angles2*(1-ratio))
    rotMat  = R.from_rotvec(angle)
    
    trans1  = pose1[:,3] 
    trans2  = pose2[:,3] 
    trans   = (trans1*ratio) + (trans2*(1-ratio))

    out[:,0:3] = rotMat.as_matrix()
    out[:,3] = trans 

    return out

def getRenderPoses(poses):
    poseSamples = 2 
    out = None 
    for i in range(poses.shape[0]-1):
        start_pose = poses[i][:,0:4] 
        end_pose   = poses[i+1][:,0:4] 
        for j in range(poseSamples):
            ratio = 1-(j/poseSamples)
            tt = getInterpolatedPose(start_pose, end_pose, ratio)
            tt = tt.reshape(1,3,4)
            if out is None:
                out = tt
            else:
                out = np.concatenate((out, tt))
    return out        


def load_llff_data(samp_img, dataset='InterHand2.6M', factor=None, recenter=True, path_zflat=False, load_mask = False, sr_factor=1, poses_bounds_fn="poses_bounds.npy"):

    poses, bds, imgs, arr_ind, distortion_mask, depths = _load_data(samp_img, factor=int(factor/sr_factor), load_mask=load_mask, poses_bounds_fn=poses_bounds_fn)
    # factor=8 downsamples original imgs by 8x
    # factor=8, sr_factor=2 downsamples original imgs by 4x
    if poses is None:
        return None, None, None, None, None
    
    if load_mask:
        mask = imgs[...,3]
        # fg_pixels = np.sum(mask == 255)
        fg_pixels = np.sum(np.logical_and(mask == 255, distortion_mask == 1)) if distortion_mask is not None else np.sum(mask == 255)
        if fg_pixels < 250:         #bad segmentation mask (too few fg pixels)
            return None, None, None, None, None
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if recenter:
        o_poses = recenter_poses(poses)
    else:
        o_poses = poses

    #select the pose corresponding to the image index
    poses = np.take(o_poses, arr_ind, 0)
    poses = poses.astype(np.float32)
    
    return images, poses, bds, distortion_mask, depths
