import numpy as np
import torch
import torch.utils.data
import os.path as osp
import os
from mano_util import create_mano_layers, get_mano_out, load_skeleton
from transform import world2cam, cam2pixel, cam2world, process_bbox
import copy
import json
from pycocotools.coco import COCO

from scipy.spatial.transform.rotation import Rotation
from tqdm import tqdm
import re
import pdb
import cv2

# mappings to familar joint order (root, then thumb to little, bottom to top)
fingertips = [745, 333, 444, 555, 672]
joint_map = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
gt_joint_map = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]

class AnnotManager:
    def __init__(self, annot_path, annot_subset, mode, smplx_path=None):
        self.annot_path = annot_path
        self.image_dir = os.path.join(annot_path, "../images")
        self.annot_subset = annot_subset
        self.mode = mode
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)

        if smplx_path is None:
            smplx_path = "./SMPLX_models/models"

        mano_layer = create_mano_layers(smplx_path, b_flat_hand_mean=False, b_Euler=False, b_root_trans=False)
        self.mano_layer = torch.nn.ModuleDict(modules=mano_layer)
        self.mano_layer.requires_grad_(False)

        mano_layer_joints = create_mano_layers(smplx_path, b_flat_hand_mean=True, b_Euler=False, b_root_trans=False)
        self.mano_layer_joints = torch.nn.ModuleDict(modules=mano_layer_joints)
        self.mano_layer_joints.requires_grad_(False)

        self.Interhand2Ours = np.array([20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16])
        self.Interhand2Ours = np.concatenate([self.Interhand2Ours, self.Interhand2Ours + 21], axis=0)

        print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset, self.mode))

        self.db = COCO(osp.join(self.annot_path, self.annot_subset, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.annot_subset, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            self.cameras = json.load(f)
        with open(osp.join(self.annot_path, self.annot_subset, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            self.joints = json.load(f)
            
        try:
            with open(osp.join(self.annot_path, self.annot_subset, self.mode,
                               'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot_with_root_offset_root_trans.json')) as f:
                self.mano_params = json.load(f)
        except:
            #load annot without offset
            with open(osp.join(self.annot_path, self.annot_subset, self.mode,
                               'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
                self.mano_params = json.load(f)
            #compute and save offset
            self.preprocess_mano()
            #load version with offset
            with open(osp.join(self.annot_path, self.annot_subset, self.mode,
                               'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot_with_root_offset_root_trans.json')) as f:
                self.mano_params = json.load(f)

        try:
            self.joint_regressor = torch.tensor(np.load("./J_regressor_mano_ih26m.npy"))
        except:
            self.joint_regressor = None

        self.info2annId = {}
        for aid, ann in self.db.anns.items():
            image_id = ann['image_id']
            img = self.db.loadImgs(image_id)[0]
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            key = '_'.join([str(capture_id), str(cam), str(frame_idx)])
            self.info2annId[key] = aid
            # pdb.set_trace()

    def get_interaction_annot_ids(self):
        try:
            return self.interaction_annots
        except AttributeError:
            filtered_annot_ids = []
            for annot_id in self.get_annot_ids():
                ann = self.get_ann(annot_id)
                if ann['hand_type'] == 'interacting':
                    filtered_annot_ids.append(annot_id)
            self.interaction_annots = filtered_annot_ids
            return self.interaction_annots
        
    def get_right_annot_ids(self):
        filtered_annot_ids = []
        for annot_id in self.get_annot_ids():
            ann = self.get_ann(annot_id)
            if ann['hand_type'] == 'right':
                filtered_annot_ids.append(annot_id)
        right_annots = filtered_annot_ids
        return right_annots


    def get_annot_id(self, capture_idx, cam_idx, frame_idx):
        key = '_'.join([str(capture_idx), str(cam_idx), str(frame_idx)])
        return self.info2annId[key]

    def get_annot_ids(self):
        return list(self.db.anns.keys())

    def get_annot_id_from_path(self, fname):
        all_nums = re.findall('\d+', fname)
        capture_idx = int(all_nums[0])
        cam_idx = int(all_nums[-2])
        frame_idx = int(all_nums[-1])
        return self.get_annot_id(capture_idx, cam_idx, frame_idx)

    def get_ann(self, aid):
        return self.db.anns[aid]

    def get_img_info(self, ann):
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]
        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']

        return str(capture_id), str(seq_name), str(cam), str(frame_idx)

    def get_img_info_more(self, ann):
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]
        capture_id = img['capture']
        subject_id = img['subject']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']

        return str(capture_id), str(seq_name), str(cam), str(frame_idx), str(subject_id)

    def get_img_fname(self, ann):
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]
        return img['file_name']
    
    def get_image(self, ann):
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]         #dict_keys(['id', 'file_name', 'width', 'height', 'capture', 'subject', 'seq_name', 'camera', 'frame_idx'])
        
        #we expect the capture_id and subject_id to be the same
        # assert img['capture'] == img['subject'], f"capture_id ({img['capture']}) and subject_id ({img['subject']}) should be the same"
        
        fname = img['file_name']
        full_path = osp.normpath(osp.join(self.image_dir, self.mode, fname))
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(full_path)
        return image
            
    def get_img_size(self, ann):
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]
        img_width, img_height = img['width'], img['height']
        return img_width, img_height

    def get_joints(self, ann):
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(self.cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

        return joint_cam, joint_img

    def get_joints_world_valid(self, ann):
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        joint_world = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joints_world_valid = np.logical_not(np.all(joint_world == 1., axis=-1))
        return joints_world_valid

    def world_joints_to_cam(self, joint_world, ann_id):
        ann = self.get_ann(ann_id)
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        return joint_cam

    def cam_joints_to_world(self, joint_cam, ann_id):
        ann = self.get_ann(ann_id)
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)

        # joint_world_tmp = world2cam(joint_cam.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        # joint_cam_orig = cam2world(joint_world_tmp.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        joint_world = cam2world(joint_cam.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)

        return joint_world


    def get_cam_RotTrans(self, ann):
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32) / 1000, np.array(
            self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        return camrot, campos

    def get_joints_valid(self, ann):
        joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
        hand_type = ann['hand_type']
        hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

        return joint_valid, hand_type, hand_type_valid

    def get_joints_from_path(self, capture_id, seq_name, cam, frame_idx):
        campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(self.cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

        return joint_cam, joint_img

    def get_cam_params(self, ann):
        capture_id, seq_name, cam, frame_idx = self.get_img_info(ann)
        return self.get_cam_params_(capture_id, cam)

    def get_cam_params_(self, capture_id, cam):
        focal, princpt = np.array(self.cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
            self.cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        return focal, princpt

    def get_bbox(self, ann):
        img_width, img_height = self.get_img_size(ann)
        bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
        bbox = process_bbox(bbox, (img_height, img_width))  # enlarge bbox (x,y,w,h) format still
        return bbox

    def get_abs_depth(self, joint_cam):
        abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                     'left': joint_cam[self.root_joint_idx['left'], 2]}
        return abs_depth

    def get_img_path(self, capture_idx, seq_name, cam_idx, frame_idx):
        img_path = os.path.join(self.image_dir, self.mode, 'Capture' + capture_idx, seq_name,
                                'cam' + cam_idx, 'image' + frame_idx + '.jpg')
        return img_path

    def get_hand_params_padded(self, hand_param):
        hand_params_padded = copy.deepcopy(hand_param)
        if hand_params_padded[0] is None:
            hand_params_padded[0] = {'pose': [0.] * 48, 'shape': [0.] * 10, 'trans': [1.] * 3, 'root_offset': [1.] * 3,
                                     'hand_type': 'right'}
        else:
            hand_params_padded[0]['hand_type'] = 'right'
        if hand_params_padded[1] is None:
            hand_params_padded[1] = {'pose': [0.] * 48, 'shape': [0.] * 10, 'trans': [1.] * 3, 'root_offset': [1.] * 3,
                                     'hand_type': 'left'}
        else:
            hand_params_padded[1]['hand_type'] = 'left'
        return hand_params_padded

    def get_hand_param_cam_coord(self, mano_param, cam_transf, hand_type):
        # get global translation and rotation (root relative)
        base_rot = mano_param['pose'][:3]
        base_trans = mano_param['trans']
        root = mano_param['root_offset']
        root = np.array(root)

        # get camera coordinate transform
        R = cam_transf['R']
        t = cam_transf['t']
        t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t

        # get global rotation and translation to root relative translation and rotation
        base_rot = Rotation.from_rotvec(base_rot)
        additional_rot = Rotation.from_matrix(R)
        final_rot = additional_rot * base_rot
        final_trans = additional_rot.apply(root) + additional_rot.apply(base_trans) - root + t

        # the fitted parameters subtracts the mano mean parameters from the hand pose. Add it back to obtain aa.
        mean_pose = self.mano_layer[hand_type].hand_mean.numpy()

        mano_param_updated = copy.deepcopy(mano_param)
        mano_param_updated['pose'][3:] = mano_param_updated['pose'][3:] + mean_pose
        mano_param_updated['pose'][:3] = final_rot.as_rotvec()
        mano_param_updated['trans'] = final_trans.tolist()
        mano_param_updated['hand_type'] = hand_type

        return mano_param_updated

    def get_hand_param_cam_coord_root_trans(self, mano_param, cam_transf, hand_type):
        # get global translation and rotation (root relative)
        base_rot = mano_param['pose'][:3]
        base_trans = mano_param['trans']

        # get camera coordinate transform
        R = cam_transf['R']
        t = cam_transf['t']
        t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t

        # get global rotation and translation to root relative translation and rotation
        base_rot = Rotation.from_rotvec(base_rot)
        additional_rot = Rotation.from_matrix(R)
        final_rot = additional_rot * base_rot
        final_trans = additional_rot.apply(base_trans) + t

        # the fitted parameters subtracts the mano mean parameters from the hand pose. Add it back to obtain aa.
        mean_pose = self.mano_layer[hand_type].hand_mean.numpy()

        mano_param_updated = copy.deepcopy(mano_param)
        mano_param_updated['pose'][3:] = mano_param_updated['pose'][3:] + mean_pose
        mano_param_updated['pose'][:3] = final_rot.as_rotvec()
        mano_param_updated['trans'] = final_trans.tolist()
        mano_param_updated['hand_type'] = hand_type

        return mano_param_updated

    def get_available_params(self, capture_idx, cam_idx, frame_idx):
        """
        converts the mano parameters
        from b_flat_hand_mean=False b_root_trans=False
        to b_flat_hand_mean=True b_root_trans=True
        """

        ann_idx = self.get_annot_id(capture_idx, cam_idx, frame_idx)
        ann = self.get_ann(ann_idx)
        # focal, princpt = self.get_cam_params(ann)
        # width, height = self.get_img_size(ann)
        #cam_param = self.get_camera_render_params(focal, princpt, width, height)

        R, t = self.get_cam_RotTrans(ann)
        cam_transf = {'R': R, 't': t}

        mano_params = self.__get_available_params_global(capture_idx, frame_idx)
        hand_types = ['right', 'left']

        mano_params_global = copy.deepcopy(mano_params)
        if not(mano_params_global[0] is None):
            mano_params_global[0]['trans'] = mano_params_global[0]['trans_root']
            del mano_params_global[0]['trans_root']

        if not(mano_params_global[1] is None):
            mano_params_global[1]['trans'] = mano_params_global[1]['trans_root']
            del mano_params_global[1]['trans_root']

        # from global to cam coordinate
        hand_param_cam_coord = []
        for hand_param, hand_type in zip(mano_params_global, hand_types):
            if hand_param is None:
                hand_param_cam_coord.append(None)
            else:
                mano_param_updated = self.get_hand_param_cam_coord_root_trans(hand_param, cam_transf, hand_type)
                hand_param_cam_coord.append(mano_param_updated)

        return hand_param_cam_coord

    def __get_available_params_global(self, capture_idx, frame_idx):
        hand_params = []

        for hand_type in ('right', 'left'):
            try:
                mano_param = self.mano_params[capture_idx][frame_idx][hand_type]
            except KeyError:
                mano_param = None
            hand_params.append(mano_param)

        return hand_params

    def preprocess_mano(self):
        mano_param_updated = copy.deepcopy(self.mano_params)
        mano_dict = mano_param_updated.items()
        for capture_idx, mano_param_frames in tqdm(mano_dict, total=len(mano_dict)):
            for frame_idx, mano_params in mano_param_frames.items():
                for hand_type in ('right', 'left'):
                    mano_param = mano_params[hand_type]
                    if mano_param is None:
                        continue
                    else:
                        # get root offset
                        base = {'pose': mano_param['pose'], 'trans': [0, 0, 0], 'shape': mano_param['shape']}
                        mano_pose = torch.FloatTensor(base['pose']).view(1, -1)
                        shape = torch.FloatTensor(base['shape']).view(1, -1)
                        trans = torch.FloatTensor(base['trans']).view(1, -1)
                        hand_render_pose = {'pose': mano_pose, 'shape': shape, "trans": trans, 'hand_type': hand_type}
                        mano_render_pose = [hand_render_pose]  # render single hand
                        mano_out = get_mano_out(mano_render_pose, self.mano_layer, bWeakPersp=False,
                                                            bMatPose=False)
                        joints = mano_out['joints']
                        joints[..., :2] *= -1 #get_verts_joints in right hand coorinate. convert back to opengl.
                        joints = joints.numpy()[0, ...]
                        root_offset = joints[0, ...]
                        mano_param['root_offset'] = list(root_offset.astype(float))

                        # get root translation
                        mano_pose = torch.FloatTensor(mano_param['pose']).view(1, -1)
                        shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                        trans = torch.FloatTensor(mano_param['trans']).view(1, -1)
                        hand_render_pose = {'pose': mano_pose, 'shape': shape, "trans": trans, 'hand_type': hand_type}
                        mano_render_pose = [hand_render_pose]  # render single hand
                        mano_out = get_mano_out(mano_render_pose, self.mano_layer, bWeakPersp=False,
                                                bMatPose=False)
                        joints = mano_out['joints']
                        mano_param['trans_root'] = torch.cat([-joints[0, 0, :2], joints[0, 0, 2:]],
                                                                       dim=-1).numpy().tolist()


        with open(osp.join(self.annot_path, self.annot_subset, self.mode,
                           'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot_with_root_offset_root_trans.json'), 'w') as f:
            json.dump(mano_param_updated, f)

    def get_mano_render_param(self, mano_params, device):
        return [{k: torch.tensor(v, device=device).float() if type(v) != str else v
                 for k,v in hand_dict.items()} for hand_dict in mano_params]

    def get_camera_render_params(self, focal, princpt, width, height):
        camera_type = 'perspective'
        camera_param = {'type': camera_type,
                        'focal_length': torch.tensor((focal[0], focal[1])).float()[None, ...],
                        'principal_point': torch.tensor((princpt[0], princpt[1])).float()[None, ...],
                        'img_dim': (width, height)}
        return camera_param


if __name__ == '__main__':

    smplx_path = "../../models"  # path to smplx models
    root_dir = "path/to/InterHand2.6M"


    annot_path = osp.join(root_dir, 'annotations')
    annot_subset = 'human'
    mode = 'test'
    manager = AnnotManager(annot_path, annot_subset, mode, smplx_path=smplx_path)
    
    pdb.set_trace()

    for ann_id in manager.get_annot_ids():
        annot = manager.get_ann(ann_id)
        fname = manager.get_img_fname(annot)
        print(fname)
        joints3D, joints2D = manager.get_joints(annot)
        break
