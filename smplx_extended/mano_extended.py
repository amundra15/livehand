from smplx import MANO, SMPL, SMPLH, FLAME, SMPLX
import os.path as osp
import torch
from typing import Optional, Tuple, Union
from torch import nn
from smplx.utils import (Struct, Tensor, MANOOutput)
# from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform, lbs
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
import pickle
import os
import numpy as np
import pdb

def batch_eulers(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        https://mathworld.wolfram.com/EulerAngles.html
        using xyz (pitch-row-yaw) convention A = BCD
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N euler angle vectors. (row, pitch, yaw)
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''


    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    pitch = rot_vecs[:, :1]
    roll = rot_vecs[:, 1:2]
    yaw = rot_vecs[:, 2:]

    # all these are (N,1)
    a11 = torch.cos(pitch) * torch.cos(yaw)
    a12 = torch.cos(pitch) * torch.sin(yaw)
    a13 = -torch.sin(pitch)
    a21 = torch.sin(roll) * torch.sin(pitch) * torch.cos(yaw) - torch.cos(roll) * torch.sin(yaw)
    a22 = torch.sin(roll) * torch.sin(pitch) * torch.sin(yaw) + torch.cos(roll) * torch.cos(yaw)
    a23 = torch.cos(pitch) * torch.sin(roll)
    a31 = torch.cos(roll) * torch.sin(pitch) * torch.cos(yaw) + torch.sin(roll) * torch.sin(yaw)
    a32 = torch.cos(roll) * torch.sin(pitch) * torch.sin(yaw) - torch.sin(roll) * torch.cos(yaw)
    a33 = torch.cos(pitch) * torch.cos(roll)

    # (N, 3, 3)
    rot_mat = torch.cat([a11, a12, a13, a21, a22, a23, a31, a32, a33], dim=1).reshape(-1, 3, 3)
    return rot_mat


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
    vertices_list: Tensor = None
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        vertices_list: indices of vertices for which the tform is returned
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if vertices_list is not None:
        extra_locs = v_posed_homo[:, vertices_list, :3]
        J = torch.cat((J, extra_locs), dim=1) #(1, 21, 3)

        extra_locs = verts[:, vertices_list, :]
        J_transformed = torch.cat((J_transformed, extra_locs), dim=1) #(1, 21, 3)

        A = torch.cat((A,T[:,vertices_list,:,:]), dim=1)    #torch.Size([1, 21, 4, 4])

    # return verts, J_transformed
    return verts, J_transformed, J, A


def lbs_euler_glob_rot(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
        vertices_list: Tensor = None
) -> Tuple[Tensor, Tensor]:
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template : torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        vertices_list: indices of vertices for which the tform is returned

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        pose_rod = pose[:, 3:]
        glob_rot_rod = pose[:, :3]
        pose_rot_mats = batch_rodrigues(pose_rod.reshape(-1, 3)).view(
            [batch_size, -1, 3, 3])

        glob_rot_mats = batch_eulers(glob_rot_rod)[:, None, ...]
        rot_mats = torch.cat([glob_rot_mats, pose_rot_mats], dim=1)  # (N, J+1, 3, 3)

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    # J_transformed, A, posedJoint2rootTransforms = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)


    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if vertices_list is not None:
        extra_locs = v_posed_homo[:, vertices_list, :3]
        J = torch.cat((J, extra_locs), dim=1) #(1, 21, 3)

        extra_locs = verts[:, vertices_list, :]
        J_transformed = torch.cat((J_transformed, extra_locs), dim=1) #(1, 21, 3)

        A = torch.cat((A,T[:,vertices_list,:,:]), dim=1)    #torch.Size([1, 21, 4, 4])
    # pdb.set_trace()

    # return verts, J_transformed, J, A, posedJoint2rootTransforms
    return verts, J_transformed, J, A



class MANOExtended(MANO):
    """
    Extended MANO class:
        - root translation parameters
        - Euler angle global rotation
        - Forward takes rotation matrix directly as input
    """

    def __init__(
            self,
            model_path: str,
            is_Euler: bool = False,
            use_root_trans: bool = False,
            scale: float = 1.0,
            **kwargs
    ) -> None:
        """ MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            is_Euler: bool, optional
                Whether the global rotation component is euler rotation or rod.
            use_root_trans: bool, optional
                Whether the translation component is the root position, or translation of the template origin.
        """
        super(MANOExtended, self).__init__(model_path, **kwargs)

        self.use_euler = is_Euler
        self.use_root_trans = use_root_trans
        
        self.scale = scale

        # redo pca dependent steps.
        self.use_pca = kwargs.get('use_pca', True) # Overwrite behavior where pca is off when using all 45 components.
        self.num_pca_comps = kwargs.get('num_pca_comps', 6)
        data_struct = kwargs.get('data_struct', None)
        ext = kwargs.get('ext', 'pkl')
        is_rhand = kwargs.get('is_rhand', True)
        dtype = kwargs.get('dtype', torch.float32)
        create_hand_pose = kwargs.get('create_hand_pose', True)
        hand_pose = kwargs.get('hand_pose', None)
        batch_size = kwargs.get('batch_size', 1)


        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(
                    model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)

            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        hand_components = data_struct.hands_components[:self.num_pca_comps]
        self.np_hand_components = hand_components
        if self.use_pca:
            self.register_buffer(
                'hand_components',
                torch.tensor(hand_components, dtype=dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = self.num_pca_comps if self.use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose,
                                           requires_grad=True)
            self.register_parameter('hand_pose',
                                    hand_pose_param)

    def forward(
            self,
            betas: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            hand_pose: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts: bool = True,
            return_full_pose: bool = False,
            use_matrix: bool = False,
            return_as_dict: bool = False,
            **kwargs
    ) -> MANOOutput:
        assert not (use_matrix and self.use_pca), "Error: Can not use both matrix input and pca pose input"
        assert not (not (
            self.flat_hand_mean) and use_matrix), "Error: can use use none flat mean hand, and matrix at the same time"

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else
                     self.hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum(
                'bi,ij->bj', [hand_pose, self.hand_components])

        if use_matrix:  # (Pose should be should have (N, J, 9)
            assert len(
                global_orient.shape) == 3, "Error: Layer configured to use rotation matrix. But rotation is not given"
            assert len(
                hand_pose.shape) == 3, "Error: Layer configured to use rotation matrix. But rotation is not given"

        full_pose = torch.cat([global_orient, hand_pose], dim=1)

        if not use_matrix:
            full_pose += self.pose_mean

        if self.use_euler:
            vertices, joints, _, _ = lbs_euler_glob_rot(betas, full_pose, self.v_template,
                                                  self.shapedirs, self.posedirs,
                                                  self.J_regressor, self.parents,
                                                  self.lbs_weights, pose2rot=True,
                                                  )
        else:
            vertices, joints, _, _ = lbs(betas, full_pose, self.v_template,
                                   self.shapedirs, self.posedirs,
                                   self.J_regressor, self.parents,
                                   self.lbs_weights, pose2rot=(not use_matrix),
                                   )
    

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            if self.use_root_trans:  # bring root (wrist) to the origin.
                joints_root_centered = joints - joints[:, :1, :]
                vertices_root_centered = vertices - joints[:, :1, :]
                joints = joints_root_centered + transl.unsqueeze(dim=1)
                vertices = vertices_root_centered + transl.unsqueeze(dim=1)
            else:
                joints = joints + transl.unsqueeze(dim=1)
                vertices = vertices + transl.unsqueeze(dim=1)
        
        vertices = vertices * self.scale
        joints = joints * self.scale


        if return_as_dict:
            output = {'betas':betas, 'global_orient':global_orient, 'hand_pose':hand_pose}
            if return_verts:
                output['vertices'] = vertices
                output['joints'] = joints
            if return_full_pose:
                output['full_pose'] = full_pose
        
        else:
            output = MANOOutput(vertices=vertices if return_verts else None,
                                joints=joints if return_verts else None,
                                betas=betas,
                                global_orient=global_orient,
                                hand_pose=hand_pose,
                                full_pose=full_pose if return_full_pose else None)

        return output


    # #adapted from forward function()
    # def get_articulation_matrices(
    #         self,
    #         betas: Optional[Tensor] = None,
    #         global_orient: Optional[Tensor] = None,
    #         hand_pose: Optional[Tensor] = None,
    #         transl: Optional[Tensor] = None,
    #         # scale: float = 1.0,
    #         return_verts: bool = True,
    #         return_full_pose: bool = False,
    #         use_matrix: bool = False,
    #         include_tips: bool = False,
    #         return_bone_centers = False):
    #         # **kwargs):
    #     '''
    #     Arguments:
    #     return_bone_centers: return the center of each bone (mid point of a joint and its parent)
    #     '''
        
    #     assert not (use_matrix and self.use_pca), "Error: Can not use both matrix input and pca pose input"
    #     assert not (not (
    #         self.flat_hand_mean) and use_matrix), "Error: can use use none flat mean hand, and matrix at the same time"
    #     assert not (return_bone_centers and not include_tips), "Error: computed bone centers currently span the whole hand only when the tips are also included"
        
    #     # If no shape and pose parameters are passed along, then use the
    #     # ones from the module
    #     global_orient = (global_orient if global_orient is not None else
    #                     self.global_orient)
    #     betas = betas if betas is not None else self.betas
    #     hand_pose = (hand_pose if hand_pose is not None else
    #                 self.hand_pose)

    #     apply_trans = transl is not None or hasattr(self, 'transl')
    #     if transl is None:
    #         if hasattr(self, 'transl'):
    #             transl = self.transl

    #     if self.use_pca:
    #         hand_pose = torch.einsum(
    #             'bi,ij->bj', [hand_pose, self.hand_components])

    #     if use_matrix:  # (Pose should be should have (N, J, 9)
    #         assert len(
    #             global_orient.shape) == 3, "Error: Layer configured to use rotation matrix. But rotation is not given"
    #         assert len(
    #             hand_pose.shape) == 3, "Error: Layer configured to use rotation matrix. But rotation is not given"

    #     full_pose = torch.cat([global_orient, hand_pose], dim=1)

    #     if not use_matrix:
    #         full_pose += self.pose_mean

    #     #this will add fingertips to the articulated joints array and also the corresponding tform matrices
    #     if include_tips:
    #         tips_indices = [745, 317, 444, 556, 673]  # thumb, index, middle, ring, little
    #     else:
    #         tips_indices = None

    #     if self.use_euler:
    #         # vertices, joints, J, posedJoint2canonicalJointTransforms, posedJoint2rootTransforms = lbs_euler_glob_rot(betas, full_pose, self.v_template,
    #         vertices, joints, J, posedJoint2canonicalJointTransforms = lbs_euler_glob_rot(betas, full_pose, self.v_template,
    #                                                                     self.shapedirs, self.posedirs,
    #                                                                     self.J_regressor, self.parents,
    #                                                                     self.lbs_weights, pose2rot=True,
    #                                                                     vertices_list = tips_indices
    #                                                                     )
    #     else:
    #         vertices, joints, J, posedJoint2canonicalJointTransforms = lbs(betas, full_pose, self.v_template,
    #                                                                         self.shapedirs, self.posedirs,
    #                                                                         self.J_regressor, self.parents,
    #                                                                         self.lbs_weights, pose2rot=(not use_matrix),
    #                                                                         vertices_list = tips_indices
    #                                                                         )

    #     if self.joint_mapper is not None:
    #         joints = self.joint_mapper(joints)

    #     if apply_trans:
    #         if self.use_root_trans:  # bring root (wrist) to the origin and then move to the desired location
    #             joints_root_centered = joints - joints[:, :1, :]
    #             vertices_root_centered = vertices - joints[:, :1, :]
    #             joints = joints_root_centered + transl.unsqueeze(dim=1)
    #             vertices = vertices_root_centered + transl.unsqueeze(dim=1)
    #         else:
    #             joints = joints + transl.unsqueeze(dim=1)
    #             vertices = vertices + transl.unsqueeze(dim=1)


    #     #J represents canonical joints
    #     #joints represents articulated joints
    #     J = J + transl.unsqueeze(dim=1)
    #     J = J[0,...] * self.scale                    #torch.Size([21, 3])
    #     joints = joints[0,...] * self.scale          #torch.Size([21, 3])
    #     vertices = vertices[0,...] * self.scale      #torch.Size([778, 3])
        
    #     bone_centers = None
    #     if return_bone_centers:
    #         parents = self.parents
    #         # print("parents: ", parents)    # tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11,  0, 13, 14],
    #         if include_tips:
    #             parents = torch.cat((parents, torch.tensor([15, 3, 6, 12, 9])))
    #             # print("parents: ", parents)    # tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11,  0, 13, 14, 15,  3, 6, 12,  9],

    #         bone_centers = (J[1:] +  J[parents[1:]])/2
    #         bone_centers = torch.cat((J[None,0], bone_centers), dim = 0)         #append the wrist location


    #     articulation_matrices = {
    #                             'canonical_joints': J,
    #                             'articulated_joints': joints,
    #                             'posedJoint2canonicalJointTransforms': posedJoint2canonicalJointTransforms,
    #                             # 'posedJoint2rootTransforms': posedJoint2rootTransforms,
    #                             # 'canonical_bone_centers': bone_centers,
    #                             'vertices': vertices,
    #                             'lbs_weights': self.lbs_weights,
    #                             }
        
    #     if return_bone_centers:
    #         articulation_matrices['canonical_bone_centers'] = bone_centers


    #     return articulation_matrices


def create(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPL, SMPLH, SMPLX, MANO, FLAME]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANOExtended(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAME(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')
