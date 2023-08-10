from itertools import count
import os.path as osp
import os
import pathlib
import pdb
import copy
import yaml
from argparse import ArgumentParser

import numpy as np
from numpy.core.numeric import zeros_like
import cv2
import pyrender
import trimesh
import smplx
import torch
import pickle
import imutils
from collections import defaultdict

from dataset_manager import AnnotManager

os.environ["PYOPENGL_PLATFORM"] = "egl"         #needed when running remotely with ssh


def allowed_annot(manager, annot, allowed_capture_ids, allowed_subject_ids, allowed_camera_ids, allowed_frame_idxs):
    
    #appropriate capture session
    capture_id = manager.get_img_info_more(annot)[0]
    if (allowed_capture_ids == 'all') or (capture_id in allowed_capture_ids):
        
        #appropriate subject
        subject_id = manager.get_img_info_more(annot)[4]
        if (allowed_subject_ids == 'all') or (subject_id in allowed_subject_ids):
            
            #appropriate cameras
            camera_id = manager.get_img_info_more(annot)[2]
            if (allowed_camera_ids == 'all') or (camera_id in allowed_camera_ids):
                
                #appropriate frame index
                frame_idx = manager.get_img_info_more(annot)[3]
                if (allowed_frame_idxs == 'all') or (frame_idx in allowed_frame_idxs):
                    
                    return True
            
    return False


def adjust_image_center(rgb, cam_param, cam_idx):
    
    im_ht, im_wt = rgb.shape[0:2]
    shift_x = cam_param['princpt'][cam_idx][0] - (im_wt/2)
    shift_y = cam_param['princpt'][cam_idx][1] - (im_ht/2)
    
    #make camera cx, cy = 0
    cam_param_updated = copy.deepcopy(cam_param)
    cam_param_updated['princpt'][cam_idx] = [im_wt/2, im_ht/2]
    #move the image center
    rgb = imutils.translate(rgb, -shift_x, -shift_y)

    return rgb, cam_param_updated


def save_mesh(mano_layer, mano_param, mesh_path):

    if osp.exists(mesh_path):
        return
    
    print("[INFO] Saving the mesh...")
    #create the folder if it doesn't exist
    dir = osp.dirname(mesh_path)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    # get MANO 3D mesh coordinates (world coordinate)
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    root_pose = mano_pose[0].view(1,3)
    hand_pose = mano_pose[1:,:].view(1,-1)
    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
    output = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy()
    mesh = trimesh.Trimesh(mesh, mano_layer.faces)
    trimesh.exchange.export.export_mesh(mesh, mesh_path)
    
    return


def save_mano_param(mano_param, filename):

    if osp.exists(filename):
        return

    print("[INFO] Saving the MANO params...")
    #create the folder if it doesn't exist
    dir = osp.dirname(filename)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    
    #covert lists to numpy arrays
    a = np.array(mano_param['pose'])
    a = a[None, ...]
    mano_param['pose'] = a

    b = np.array(mano_param['shape'])
    b = b[None, ...]
    mano_param['shape'] = b

    c = np.array(mano_param['trans'])
    c = c[None, ...]
    mano_param['trans'] = c

    with open(filename, 'wb') as f:
        pickle.dump(mano_param, f)
    
    return



def save_camera(fs, cam_param, cam_idx, cam_indices_map):
#taken from render_mano_mask()
    
    print("[INFO] Saving the camera params...")
    # add camera extrinsics
    t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
    t = t/1000      #convert mm to m
    Rt = np.concatenate((np.linalg.inv(R),np.transpose(t[None])),axis=1)       #shape: [3,4]
    fs.write("extrinsic-"+str(cam_indices_map[cam_idx]), Rt)
        
    # add camera intrinsics
    focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
    princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
    intr = np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]], dtype=np.float32)
    fs.write("intrinsic-"+str(cam_indices_map[cam_idx]), intr)
    
    return


def render_mano_mask(mano_layer, mano_param, cam_param, cam_idx, img_height, img_width):
#taken from InterHands2.6M render.py

    prev_depth = None
    
    # get MANO 3D mesh coordinates (world coordinate)
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    root_pose = mano_pose[0].view(1,3)
    hand_pose = mano_pose[1:,:].view(1,-1)
    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
    output = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000 # meter to milimeter

    # apply camera extrinsics
    t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
    t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
    mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    
    # mesh
    mesh = mesh / 1000 # milimeter to meter
    mesh = trimesh.Trimesh(mesh, mano_layer.faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
            
    # add camera intrinsics
    focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
    princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
    
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    depth = depth[:,:,None]
    valid_mask = (depth > 0)
    if prev_depth is None:
        render_mask = valid_mask
        prev_depth = depth
    else:
        render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth==0)
        prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

    #convert binary to uint8
    render_mask = render_mask.astype(np.uint8) * 255

    return render_mask


def grabcut_refine(mano_mask, rgb_img):
                
    #erode the MANO mask
    kernel = np.ones((25,25),np.uint8)
    mano_mask_eroded = cv2.erode(mano_mask*255, kernel, iterations=1)
            
    grabCut_mask = zeros_like(mano_mask)
    grabCut_mask[mano_mask_eroded > 0] = cv2.GC_PR_FGD
    grabCut_mask[mano_mask_eroded == 0] = cv2.GC_PR_BGD

    #GRABCUT
    # allocate memory for two arrays that the GrabCut algorithm internally uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the the mask segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(rgb_img, grabCut_mask, None, bgModel, fgModel, iterCount=20, mode=cv2.GC_INIT_WITH_MASK)

    # set all definite background and probable background pixels to 0 while definite foreground and probable foreground pixels are set to 1, then scale teh mask from the range [0, 1] to [0, 255]
    refined_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD) , 0, 1)
    refined_mask = (refined_mask * 255).astype("uint8")
    refined_mask = refined_mask[...,0]

    return refined_mask
    

def largest_component(mask):
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])		# Note: range() starts from 1 since 0 is the background label.
    finalMask = zeros_like(mask)
    finalMask[labels == max_label] = 255
    return finalMask


def remove_forearm(mano_mask, mano_mask_refined):
    
    kernel = np.ones((10,10),np.uint8)
    mano_mask_dilated = cv2.dilate(mano_mask, kernel, iterations=1)
    _, diff = cv2.threshold(mano_mask_refined - mano_mask_dilated, 127, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(diff) == 0:         #mano_mask_dilated encapsulates the mano_mask_refined; nothing to remove
        return mano_mask_refined
    
    probable_forearm = largest_component(diff)
    #estimate mask area
    mask_area_frac = cv2.countNonZero(probable_forearm)/(mano_mask.shape[0]*mano_mask.shape[1])
    
    if mask_area_frac > 0.01:
        #extra region big enough to be a forearm
        return mano_mask_refined - probable_forearm
    else:
        #its probably some part of the palm
        return mano_mask_refined
    
    
def append_mask(rgb, mask):
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba

def imfuse(im1, im2):
    #MATLAB style color fuse
    result = np.dstack((im1, cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), im1))
    return result

    
if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("--job_array", dest="job_array", type=bool, default=False, help="Whether the script is called as a job array")
    parser.add_argument("--cpu_index", dest="cpu_index", type=int, default=0, help="The index of the CPU array")
    parser.add_argument("--input_dir", dest="input_dir", type=str, default=None, help="The input directory where the downloaded InterHands2.6M dataset is stored")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=None, help="The output directory where the processed dataset will be stored")
    args = parser.parse_args()
    
    print(f"[INFO] prepare_dataset() called with CPU index: {args.cpu_index}")
    input_dir = args.input_dir
    output_dir = args.output_dir
    assert osp.exists(input_dir), f"Input directory {input_dir} does not exist"
    pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True)
    
    smplx_path = "../../models"  # path to smplx models


    #select the data subset to process based on InterHand2.6M labels
    annot_subset = 'all'        #'human'
    mode = 'test'               #'train'
    
    allowed_capture_ids = ['0']
    assert len(allowed_capture_ids) == 1, "single element in allowed_capture_ids for now"
    allowed_subject_ids = 'all'
    with open("./cam_indices_140cams.yaml", "r") as stream:
        cam_indices_map = yaml.safe_load(stream)
    allowed_camera_ids = list(cam_indices_map.keys())
    allowed_frame_idxs = 'all'


    #load annotations
    annot_path = osp.join(input_dir, 'annotations')
    manager = AnnotManager(annot_path, annot_subset, mode, smplx_path=smplx_path)
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 
                'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}


    #camera param file objects
    if args.cpu_index == 0:         #save only with CPU 0 (so that they dont overwrite each other) NOTE: works only with a single capture id
        fss = {}
        saved_cam_params = []
        for capture_id in allowed_capture_ids:
            cameraYML = osp.join(output_dir, 'calib.yml')
            fs = cv2.FileStorage(cameraYML, cv2.FILE_STORAGE_WRITE)
            fss[capture_id] = fs

    valid_masks = {}

    #collate all the frames to be processed
    frames_to_process = defaultdict(list)
    count = 0
    for ann_id in manager.get_right_annot_ids():
        annot = manager.get_ann(ann_id)
        if allowed_annot(manager, annot, allowed_capture_ids, allowed_subject_ids, allowed_camera_ids, allowed_frame_idxs):
            cam_idx = manager.get_img_info_more(annot)[2]
            frame_id = manager.get_img_info_more(annot)[3]
            frames_to_process[frame_id].append(cam_idx)
            count += 1
    frames_to_process = dict(frames_to_process)      
    print(f"{len(frames_to_process)} relavant frames_ids, {count} relavant annot_ids found in the annotation file")
    
    
    #retrieve the frames for the current CPU
    if args.job_array:                  #called from run_prepare_dataset.sh as a job array        
        frames_to_process_curr_cpu = list(frames_to_process.items())[args.cpu_index:args.cpu_index+1]
    else:
        frames_to_process_curr_cpu = list(frames_to_process.items())
    print(f"Current CPU will process {len(frames_to_process_curr_cpu)} frames")
    
    
    for frame_id, cam_idxs in frames_to_process_curr_cpu:
        
        #eg: frame_id = '23330'
        # cam_idxs = ['400262', '400263', '400264', '400265', '400266', '400267', '400268', '400269', ...]
    
        for cam_idx in cam_idxs:
            
            ann_id = manager.get_annot_id(allowed_capture_ids[0], cam_idx, frame_id)        #NOTE: hard-coded to first capture_id here
            annot = manager.get_ann(ann_id)
            fname = manager.get_img_fname(annot)
            print(fname)
        
            #retrive the meta info
            capture_id = manager.get_img_info_more(annot)[0]
            subject_id = manager.get_img_info_more(annot)[4]
            cam_idx = manager.get_img_info_more(annot)[2]
            frame_id = manager.get_img_info_more(annot)[3]
            
            #create the directory structure
            pose_dir = osp.join(output_dir, f'frame{int(frame_id):05}')
            pathlib.Path(pose_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(osp.join(pose_dir, 'images')).mkdir(parents=True, exist_ok=True)
            
            #retrive the info
            mano_param = manager.mano_params[capture_id][frame_id]['right']
            cam_param = manager.cameras[capture_id]
            img_width, img_height = manager.get_img_size(annot)
            rgb = manager.get_image(annot)
            
            
            #nerf code expects camera cx, cy to be 0
            rgb, cam_param = adjust_image_center(rgb, cam_param, cam_idx)

            #generate mask with MANO projection
            mano_mask = render_mano_mask(mano_layer['right'], mano_param, cam_param, cam_idx, img_height, img_width)

            #refine mask with grabcut
            mano_mask_gc = grabcut_refine(mano_mask, rgb)
            
            #largest component
            mano_mask_gc_lc = largest_component(mano_mask_gc)
            
            #detect forearm region in the mask (if any)
            mano_mask_forearm_removed = remove_forearm(mano_mask, mano_mask_gc_lc)
            
            # append the mask to the rgb image
            rgba = append_mask(rgb, mano_mask_forearm_removed)
            cv2.imwrite(osp.join(pose_dir, 'images', f'{cam_indices_map[cam_idx]}.png'), rgba)

            #save mano_mesh (if it doesn't exist)
            save_mesh(mano_layer['right'], mano_param, mesh_path=osp.join(pose_dir,'mesh','mesh.obj'))

            #save mano_param (if it doesn't exist)
            mano_param['hand_type'] = 'right'
            save_mano_param(mano_param, osp.join(pose_dir, 'mesh', 'MANO_params.pkl'))


            #NeRF code expects the images to have cx=cy=0; thus we shift the images and save a valid mask
            valid_mask = np.ones((img_height, img_width))
            updated_valid_mask, _ = adjust_image_center(valid_mask, cam_param, cam_idx)
            valid_masks[cam_indices_map[cam_idx]] = updated_valid_mask.astype(bool)


            #save cameras
            if 'fss' in locals():
                if f'{capture_id}_{cam_idx}' not in saved_cam_params:
                    save_camera(fss[capture_id], cam_param, cam_idx, cam_indices_map)
                    saved_cam_params.append(f'{capture_id}_{cam_idx}')


        if 'fss' in locals():
            for fs in fss.values():    
                fs.release()
        
        #save the masks
        np.save(os.path.join(output_dir, 'valid_pixel_masks.npy'), valid_masks)


    print(f"[INFO] Done with prepare_dataset() for CPU index {args.cpu_index}")