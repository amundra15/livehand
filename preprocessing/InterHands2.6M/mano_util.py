import torch
import numpy as np
import sys

sys.path.append("../../")
import smplx_extended

def create_mano_layers(smplx_path, b_flat_hand_mean, b_Euler, b_root_trans):
    mano_layer = {'right': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=True,
                                                 flat_hand_mean=b_flat_hand_mean,
                                                 num_pca_comps=45, is_Euler=b_Euler, use_root_trans=b_root_trans),
                  'left': smplx_extended.create(smplx_path, 'mano', use_pca=False, is_rhand=False,
                                                flat_hand_mean=b_flat_hand_mean,
                                                num_pca_comps=45, is_Euler=b_Euler, use_root_trans=b_root_trans)}
    if torch.sum(
            torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1
    return mano_layer


def mano_ortho2perspective(params_ortho, cam_param):
    assert len(params_ortho) == 2, "Error. orthographic rendering only supported for rendering 2 hands"

    focal_lengths = cam_param['focal_length']  # (N, 2)
    f = focal_lengths.mean(dim=-1)[..., None]  # (N, 1)

    princpt = cam_param['principal_point']

    uv0 = params_ortho[0]['uv']
    scale0 = params_ortho[0]['scale']

    xy0 = (uv0 - princpt) / scale0
    z0 = f / scale0
    pos0 = torch.cat([xy0, z0], dim=-1)

    uv1 = params_ortho[1]['uv']
    scale1 = params_ortho[1]['scale']
    xy1 = (uv1 - princpt) / scale1
    z1 = f / scale1
    pos1 = torch.cat([xy1, z1], dim=-1)

    params = [{k: v.clone() if type(v) == torch.Tensor else v for k, v in param.items()} for param in params_ortho]
    params[0]['trans'] = pos0
    params[1]['trans'] = pos1

    del params[0]['uv']
    del params[1]['uv']
    del params[0]['scale']
    del params[1]['scale']

    return params


def mano_perspective2ortho(params, cam_param):
    assert len(params) == 2, "Error. orthographic rendering only supported for rendering 2 hands"

    focal_lengths = cam_param['focal_length']  # (N, 2)
    f = focal_lengths.mean(dim=-1)[..., None]  # (N, 1)
    princpt = cam_param['principal_point']

    trans0 = params[0]['trans']
    xy0 = trans0[..., :2]
    z0 = trans0[..., 2:3]

    trans1 = params[1]['trans']
    xy1 = trans1[..., :2]
    z1 = trans1[..., 2:3]

    scale0 = f / z0
    scale1 = f / z1
    uv0 = scale0 * xy0 + princpt
    uv1 = scale1 * xy1 + princpt

    params = [{k: v.clone() if type(v) == torch.Tensor else v for k, v in param.items()} for param in params]

    params[0]['uv'] = uv0
    params[0]['scale'] = scale0
    params[1]['uv'] = uv1
    params[1]['scale'] = scale1

    del params[0]['trans']
    del params[1]['trans']

    return params


def get_mano_out(params, mano_layer, bWeakPersp=False, bMatPose=False):
    """
    Args:
        params: mano parameters
        mano_layer: mano layer used to transform the parameters
        bWeakPersp: whether params are given with weak perspective translation or global translation
        bMatPose: whether articulation is mano parameters or rotation matrix
    Returns:

    """
    all_vertices = []
    all_centers = []
    all_stddevs = []
    all_joints = []
    hand_types = []
    MANO2MANUS_joint_map = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    MANUS2MANO_joint_map = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 5, 16, 20]

    tips = [745, 333, 444, 555, 672]  # thumb, index, middle, ring, little (ours)
    # tips = [745, 317, 445, 556, 673]  # thumb, index, middle, ring, little (Hasson left)
    # tips = [745, 317, 444, 556, 673]  # thumb, index, middle, ring, little (Hasson right)


    if (bWeakPersp):
        for param_id in range(len(params)):
            param = params[param_id]
            hand_type = param['hand_type']

            vertices, _, joints_mano = _get_MANO_mesh_weak_perspective(param, hand_type, mano_layer, bMatPose)
            joints_mano = torch.cat((joints_mano, vertices[:, tips, :]), dim=1)
            joints = joints_mano[:, MANO2MANUS_joint_map, :]

            all_vertices.append(vertices)
            all_joints.append(joints)
            hand_types.append(hand_type)
    else:
        for param_id in range(len(params)):
            param = params[param_id]
            hand_type = param['hand_type']

            vertices, _, joints_mano = _get_MANO_mesh(param, hand_type, mano_layer, bMatPose)
            joints_mano = torch.cat((joints_mano, vertices[:, tips, :]), dim=1)
            joints = joints_mano[:, MANO2MANUS_joint_map, :]

            all_vertices.append(vertices)
            all_joints.append(joints)
            hand_types.append(hand_type)

    all_vertices = torch.cat(all_vertices, dim=1)
    all_joints = torch.cat(all_joints, dim=1)  # (N, 2*J, 3)

    all_joints[..., :2] *= -1
    all_vertices[..., :2] *= -1

    out_dict = {
        "verts": all_vertices,
        "joints": all_joints,
    }

    return out_dict


def regress_joints(joint_regressor, vertices):
    Interhand2Ours = np.array([20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16])
    Interhand2Ours = np.concatenate([Interhand2Ours, Interhand2Ours + 21], axis=0)
    joints_right = torch.tensordot(joint_regressor, vertices[:, :778, :], dims=[[1], [1]])
    joints_right = joints_right.permute([1, 0, 2])
    joints_left = torch.tensordot(joint_regressor, vertices[:, 778:, :], dims=[[1], [1]])
    joints_left = joints_left.permute([1, 0, 2])
    joints = torch.cat([joints_right, joints_left], dim=1)
    joints = joints[:, Interhand2Ours, :]
    return joints


def _get_MANO_mesh_weak_perspective(mano_param, hand_type, mano_layer, bMatPose=False):
    mano_pose = mano_param['pose']
    shape = mano_param['shape']
    scale = mano_param['scale']
    uv = mano_param['uv']

    if mano_pose.ndim == 1:
        mano_pose = mano_pose.view(1, len(mano_pose))
    if shape.ndim == 1:
        shape = shape.view(1, len(shape))
    if scale.ndim == 1:
        scale = scale.view(1, len(scale))  # (N, 1)
    if uv.ndim == 1:
        uv = uv.view(1, len(uv))  # (N, 2)

    trans = torch.cat([uv, 1. / scale * 1000 + 1000], dim=-1)  # (N, 3)
    mano_param['trans'] = trans

    mesh, faces, joints = _get_MANO_mesh(mano_param, hand_type, mano_layer, bMatPose)


    # apply weak perspective scaling of meshes. (To be rendered in an orthographic camera)
    mesh = scale * (mesh - trans) + trans
    joints = scale * (joints - trans) + trans

    return mesh, faces, joints


def _get_MANO_mesh(mano_param, hand_type, mano_layer, bMatPose=False):
    """
    assume inputs are already tensors
    pose: (N,48) or (48)
    trans: (N, 3) or (3)
    """

    shape = mano_param['shape']
    trans = mano_param['trans']

    if bMatPose:
        mano_pose = mano_param['rot_mat']
        if mano_pose.ndim == 3:
            mano_pose = mano_pose[None, ...]
        if shape.ndim == 1:
            shape = shape[None, ...]
        if trans.ndim == 1:
            trans = trans[None, ...]

        mano_pose = mano_pose.flatten(-2)
        root_pose = mano_pose[:, :1, :]
        hand_pose = mano_pose[:, 1:, :]

    else:
        mano_pose = mano_param['pose']
        if mano_pose.ndim == 1:
            mano_pose = mano_pose[None, ...]
        if shape.ndim == 1:
            shape = shape[None, ...]
        if trans.ndim == 1:
            trans = trans[None, ...]

        mano_pose = mano_pose.view(mano_pose.shape[0], -1, 3)
        root_pose = mano_pose[:, :1, :].view(mano_pose.shape[0], -1)
        hand_pose = mano_pose[:, 1:, :].view(mano_pose.shape[0], -1)

    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans,
                                   use_matrix=bMatPose)
    mesh = output.vertices  # meter unit
    joints = output.joints  # meter uni
    faces = mano_layer[hand_type].faces_tensor

    return mesh, faces, joints


def close_MANO(faces, vert_colors, hand_type):
    if hand_type == 'right':
        closure = np.array([[80, 79, 215],
                            [122, 215, 79],
                            [215, 216, 80],
                            [216, 280, 120],
                            [280, 240, 119],
                            [119, 240, 123],
                            [240, 235, 123],
                            [235, 93, 123],
                            [93, 39, 123],
                            [119, 118, 280],
                            [118, 120, 280],
                            [120, 121, 109],
                            [216, 120, 109],
                            [109, 80, 216]])
        closure = closure - 1  # one index to zero index

    elif hand_type == 'left':
        closure = np.array([[240, 280, 119],
                            [216, 215, 80],
                            [215, 122, 79],
                            [80, 215, 79],
                            [80, 109, 216],
                            [109, 121, 120],
                            [216, 109, 120],
                            [120, 118, 280],
                            [216, 120, 280],
                            [118, 119, 280],
                            [119, 123, 240],
                            [123, 39, 93],
                            [93, 235, 123],
                            [235, 240, 123]])
        closure = closure - 1  # one index to zero index

    vert_ids = np.sort(np.unique(closure))  # same set of vertex_ids from left and right
    new_ids = np.arange(len(vert_ids)) + 778
    closure_new = closure.copy()
    for i in range(len(vert_ids)):
        closure_new[closure == vert_ids[i]] = new_ids[i]

    closure_new = torch.tensor(closure_new, device=faces.device)
    faces_new = torch.cat([faces, closure_new], dim=0)
    color_new = torch.cat([vert_colors, torch.ones((new_ids.shape[0], 3), device=vert_colors.device)], dim=0)

    return vert_ids, faces_new, color_new

def get_faces(mano_params, mano_layer):
    """

    Args:
        mano_params: list of hand dicts
        mano_layer: mano layer
    Returns:
        merged_faces: (F, 3)
    """
    faces_list = []
    offset = 0
    for hand_param in mano_params:
        hand_type = hand_param['hand_type']
        face_tensor = mano_layer[hand_type].faces_tensor
        faces_list.append(face_tensor + offset)
        offset += face_tensor.max() + 1
    merged_faces = torch.cat(faces_list, dim=0)
    return merged_faces

def vis_mano_mesh3D(params, mano_layer, bShowGauss=True):
    import trimesh
    mano_out = get_mano_out(params, mano_layer) #(N, num_hands*verts, 3)
    all_vertices, all_gaussians = mano_out['verts'], mano_out['gaussians']

    all_faces = get_faces(params, mano_layer)[None,...] #(1, num_hands*num_faces, 3)


    mesh = trimesh.Trimesh(vertices=all_vertices[0,...].detach().cpu().numpy(),
                           faces=all_faces[0,...].detach().cpu().numpy())

    if bShowGauss:
        combined = [mesh]
        centers, stddevs = all_gaussians
        gaussians = []
        num_gauss = centers.shape[1]
        centers = centers.detach().cpu().numpy()[0, :, :]
        stddevs = stddevs.detach().cpu().numpy()[0, :, 0]
        for i in range(num_gauss):
            sphere_mesh = trimesh.primitives.Sphere(radius=stddevs[i], center=centers[i], subdivisions=3)
            gaussians.append(sphere_mesh)
        combined = combined + gaussians
        mesh = trimesh.util.concatenate(combined)
    mesh.show()

def load_skeleton(path, joint_num):
    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton

if __name__ == '__main__':
    smplx_path = "X:/HandFlow/static00/ProMANO/data"
    mano_layer = create_mano_layers(smplx_path, b_flat_hand_mean=False, b_Euler=False, b_root_trans=True)
    mano_param = [{'pose': torch.zeros(1, 48), 'shape': torch.zeros(1, 10),
                   'trans': torch.zeros(1, 3), 'hand_type':'left'}]
    vis_mano_mesh3D(mano_param, mano_layer)

    mano_param = [{'pose': (torch.rand((1, 48))-0.5)*2, 'shape': (torch.rand((1, 10))-0.5)*2,
                   'trans': (torch.rand((1, 3))-0.5)*2, 'hand_type':'left'}]
    vis_mano_mesh3D(mano_param, mano_layer)