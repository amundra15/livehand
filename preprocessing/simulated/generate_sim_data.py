from itertools import count
import os.path as osp
import os
import pathlib
import pdb
import sys
import yaml
from argparse import ArgumentParser
import copy
import pickle

import numpy as np
import cv2
import pyrender
import trimesh
import smplx
import torch
import imutils
from collections import defaultdict

os.environ["PYOPENGL_PLATFORM"] = "egl"         #needed when running remotely with ssh


sys.path.append('../InterHands2.6M')
from dataset_manager import AnnotManager
from prepare_dataset import allowed_annot, append_mask, save_mesh, save_mano_param, save_camera




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


def fix_camera_intrinsics(cam_param, cam_idx, im_wt, im_ht):
    
    # pdb.set_trace()
    
    #make camera cx, cy = 0
    cam_param_updated = copy.deepcopy(cam_param)
    cam_param_updated['princpt'][cam_idx] = [im_wt/2, im_ht/2]
    
    #make fy = fx
    cam_param_updated['focal'][cam_idx][1] = cam_param_updated['focal'][cam_idx][0]

    return cam_param_updated



def fix_mesh_shape(mesh, vt, f, ft):
    
    '''
    Add missing vertices to the mesh such that it has the same number of vertices as the texture coordinates
    mesh: 3D vertices of the orginal mesh
    vt: 2D vertices of the texture map
    f: 3D faces of the orginal mesh (0-indexed)
    ft: 2D faces of the texture map (0-indexed)
    '''

    #build a correspondance dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondances = {}
    
    #traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondances:
            correspondances[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondances[f_flat[i]]:
                correspondances[f_flat[i]].append(ft_flat[i])
    
    #build a mesh using the texture map vertices
    new_mesh = np.zeros((vt.shape[0], 3))
    for old_index, new_indices in correspondances.items():
        for new_index in new_indices:
            new_mesh[new_index] = mesh[old_index]
    
    #define new faces using the texture map faces
    f_new = ft
    
    return new_mesh, f_new



def render_mano(mano_layer, mano_param, cam_param, cam_idx, img_height, img_width):
#taken from InterHands2.6M render.py

    # # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    # if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    #     print('Fix shapedirs bug of MANO')
    #     mano_layer['left'].shapedirs[:,0,:] *= -1
                

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
    
    
    uv_coords = pickle.load(open('../../models/mano/uvs_right.pkl', 'rb'))
    #needed when run pyrender.Mesh.from_trimesh(mesh, smooth=False)
    mesh, f = fix_mesh_shape(mesh, uv_coords['verts_uvs'], mano_layer.faces, uv_coords['faces_uvs'])           #make the number of 3D vertices and texture map vertices the same


    # mesh
    mesh = mesh / 1000 # milimeter to meter
    # mesh = trimesh.Trimesh(mesh, mano_layer.faces, process=False)
    mesh = trimesh.Trimesh(mesh, f, process=False)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    
    #TEXTURE    
    # A. add default texture
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    # mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    # B. add custom texture from file
    # colors = np.loadtxt('dense_corrs.txt')
    # mesh.visual.vertex_colors = colors[:,:3]
    # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # C. add random texture
    np.random.seed(0)
    mesh.visual.face_colors = np.random.uniform(size=mesh.faces.shape)
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # D. apply using a uv map
    # texture_image = cv2.imread('tex_im.png')
    # material = trimesh.visual.texture.SimpleMaterial(image=texture_image)
    # color_visuals = trimesh.visual.TextureVisuals(uv=uv_coords['verts_uvs'], image=texture_image, material=material)
    # mesh.visual = color_visuals
    # pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(pyr_mesh, 'mesh')

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

    # append the mask to the rgb image
    rgba = append_mask(rgb, np.squeeze(render_mask))
        
    return rgba




    
if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("--cpu_index", dest="cpu_index", type=int, default=0, help="The index of the CPU array")
    args = parser.parse_args()
    print(f"[INFO] generate_sim_data() called with CPU index: {args.cpu_index}")
    
    smplx_path = "../../models"  # path to smplx models
    root_dir_src = "path/to/InterHand2.6M/dataset"
    root_dir_dst = "save/path"
    pathlib.Path(root_dir_dst).mkdir(parents=True, exist_ok=True)

    annot_path = osp.join(root_dir_src, 'annotations')
    annot_subset = 'all'        #'human'
    mode = 'test'               #'train'
    manager = AnnotManager(annot_path, annot_subset, mode, smplx_path=smplx_path)
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 
                'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    
    allowed_capture_ids = ['0']
    allowed_subject_ids = ['10']        #NOTE: for now, this is also hard-coded in line 182
    assert len(allowed_capture_ids) == 1, "single element in allowed_capture_ids for now"
    assert len(allowed_subject_ids) == 1, "single element in allowed_subject_ids for now"
    
    with open("../InterHands2.6M/cam_indices_140cams.yaml", "r") as stream:
        cam_indices_map = yaml.safe_load(stream)
    allowed_camera_ids = list(cam_indices_map.keys())

    allowed_frame_idxs = 'all' 
    
    #camera param files
    #NOTE hard-coded subject id below for the moment
    if args.cpu_index == 0:         #save only with CPU 0 (so that they dont overwrite each other) NOTE: works only with a single capture id
        fss = {}
        saved_cam_params = []
        for capture_id in allowed_capture_ids:
            capture_dir = osp.join(root_dir_dst, f'capture{capture_id}_subject10_{mode}_{annot_subset}Annot')
            pathlib.Path(capture_dir).mkdir(parents=True, exist_ok=True)
            cameraYML = osp.join(capture_dir, 'calib.yml')
            fs = cv2.FileStorage(cameraYML, cv2.FILE_STORAGE_WRITE)
            fss[capture_id] = fs


    #collate all the frames to be processed
    frames_to_process = defaultdict(list)
    count = 0
    for ann_id in manager.get_right_annot_ids():
        annot = manager.get_ann(ann_id)
        if allowed_annot(manager, annot, allowed_capture_ids, allowed_subject_ids, allowed_camera_ids, allowed_frame_idxs):         #these conditions should ideally be put while creating the object. what say?
            cam_idx = manager.get_img_info_more(annot)[2]
            frame_id = manager.get_img_info_more(annot)[3]
            frames_to_process[frame_id].append(cam_idx)
            count += 1
    frames_to_process = dict(frames_to_process)      
    print(f"{len(frames_to_process)} relavant frames_ids, {count} relavant annot_ids found in the annotation file")
    
    
    # for ann_id in allowed_annot_ids:          #to run sequentially from directly this file
    #retrieve the frames for the current CPU
    frame_id, cam_idxs = list(frames_to_process.items())[args.cpu_index]       #to run in parallel using run_prepare_dataset.sh
    
    
    
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
        pose_dir =  osp.join(root_dir_dst, f'frame{int(frame_id):05}')
        pathlib.Path(pose_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(osp.join(pose_dir, 'images')).mkdir(parents=True, exist_ok=True)
        
        #retrive the info
        mano_param = manager.mano_params[capture_id][frame_id]['right']
        mano_param['shape'] = [0]*10            #explicitly set the mano shape parameter for all poses of a given identity (dataset fits MANO independently for each pose)
        cam_param = manager.cameras[capture_id]
        img_width, img_height = manager.get_img_size(annot)
        # rgb = manager.get_image(annot)
        
        #nerf code expects camera cx, cy to be 0
        # rgb, cam_param = adjust_image_center(rgb, cam_param, cam_idx)
        cam_param = fix_camera_intrinsics(cam_param, cam_idx, img_width, img_height)
        
        #generate mask with MANO projection
        mano_rgb = render_mano(mano_layer['right'], mano_param, cam_param, cam_idx, img_height, img_width)
        cv2.imwrite(osp.join(pose_dir, 'images', f'{cam_indices_map[cam_idx]}.png'), mano_rgb)
        # cv2.imwrite(osp.join(pose_dir, 'mesh/dense_corr', f'{cam_indices_map[cam_idx]}.png'), mano_rgb)
        
        
        #save mano_mesh (if it doesn't exist)
        save_mesh(mano_layer['right'], mano_param, mesh_path=osp.join(pose_dir,'mesh','mesh.obj'))

        #save mano_param (if it doesn't exist)
        mano_param['hand_type'] = 'right'
        save_mano_param(mano_param, osp.join(pose_dir, 'mesh', 'MANO_params.pkl'))

        #save cameras
        if 'fss' in locals():
            if f'{capture_id}_{cam_idx}' not in saved_cam_params:
                save_camera(fss[capture_id], cam_param, cam_idx, cam_indices_map)
                saved_cam_params.append(f'{capture_id}_{cam_idx}')


    if 'fss' in locals():
        for fs in fss.values():    
            fs.release()


    print(f"[INFO] Done with prepare_dataset() for CPU index {args.cpu_index}")