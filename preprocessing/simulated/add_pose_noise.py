import torch
import sys
import pdb

sys.path.append('../InterHand2.6M')
from mano_util import get_mano_out, create_mano_layers


#NOTE: the dependencies are not complete


def get_mano_pose(joints_flat, joints_posed, return_cum=False):
    """

    Args:
        joints_flat: #(N, J, 3)
        joints_posed: #(N, J, 3)

    Returns:
        axis angle rep of relative rotations
    """
    # assume joints are povided in our coordinates
    # need to flip x and y
    joints_flat = torch.clone(joints_flat)
    joints_flat[:, :, :2] *= -1

    joints_posed = torch.clone(joints_posed)
    joints_posed[:, :, :2] *= -1
    #urs2mano_rot = torch.tensor([0, 1, 2, 3, 4, 5, 6,  7,  8,  9, 10, 11, 12,13,14,15])
    ours2mano_rot = torch.tensor([0, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 1, 2, 3])
    mano2ours_rot = torch.tensor([0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9])

    # remove roots
    joints_flat_right_normed = joints_flat[:, :21, :] - joints_flat[:, :1, :]
    # joints_flat_left_normed = joints_flat[:, 21:, :] - joints_flat[:, 21:22, :]
    joints_posed_right_normed = joints_posed[:, :21, :] - joints_posed[:, :1, :]
    # joints_posed_left_normed = joints_posed[:, 21:, :] - joints_posed[:, 21:22, :]

    new_poses = []
    all_cum_rot = []
    # for hand_joints_flat, hand_joints_posed in zip([joints_flat_right_normed, joints_flat_left_normed],
    #                             [joints_posed_right_normed, joints_posed_left_normed]):

    #     partial_rots_right, _, cum_rots = align_hand(hand_joints_flat, hand_joints_posed)

    #     #joint_errors = torch.norm(hand_joints_flat_partial_aligned - hand_joints_posed, dim=-1)
    #     #assert torch.all(joint_errors < 0.001), "Error: can't find best rotation"

    #     partial_rots = torch.stack(partial_rots_right, dim=1)[:,ours2mano_rot, ...] #(N, J, 3, 3)
    #     batch_size = partial_rots.shape[0]
    #     aa_rep = quaternion_to_axis_angle(matrix_to_quaternion(partial_rots.flatten(0,1))).reshape(batch_size, 48) #(N, 3, 3)
    #     new_poses.append(aa_rep)
    #     cum_rots = torch.stack(cum_rots, dim=1)
    #     all_cum_rot.append(cum_rots)
        
        
    partial_rots_right, _, cum_rots = align_hand(joints_flat_right_normed, joints_posed_right_normed)

    #joint_errors = torch.norm(hand_joints_flat_partial_aligned - hand_joints_posed, dim=-1)
    #assert torch.all(joint_errors < 0.001), "Error: can't find best rotation"

    partial_rots = torch.stack(partial_rots_right, dim=1)[:,ours2mano_rot, ...] #(N, J, 3, 3)
    batch_size = partial_rots.shape[0]
    aa_rep = quaternion_to_axis_angle(matrix_to_quaternion(partial_rots.flatten(0,1))).reshape(batch_size, 48) #(N, 3, 3)
    new_poses.append(aa_rep)
    cum_rots = torch.stack(cum_rots, dim=1)
    all_cum_rot.append(cum_rots)
        
        

    if return_cum:
        return new_poses, all_cum_rot
    else:
        return new_poses



def add_hand_pose_noise(mano_layer_ori, root_pose, hand_pose, shape, trans, hand_type, noise_seed=0, noise_std=0.001, mano_layer_fitting=None):
    
    #noise_std = 0.001: gaussian with a standard deviation of 1 mm along each axis
    
    #compensate for flat_hand_mean = False
    if mano_layer_ori[hand_type].flat_hand_mean == False:
        hand_pose = hand_pose + mano_layer_ori[hand_type].pose_mean[3:]
        # root_pose = root_pose + mano_layer_ori[hand_type].pose_mean[:3]     #experimental
    ##? Question: do we need to add to the root pose too?
    
    

    if mano_layer_fitting is None:
        # define a new MANO layer with flat_hand_mean = True
        b_flat_hand_mean = True
        b_Euler = False
        b_root_trans = True
        smplx_path = "../../models"  # path to smplx models
        mano_layer_fitting = create_mano_layers(smplx_path, b_flat_hand_mean, b_Euler, b_root_trans)


    # get a flat pose
    mano_pose_right = {'pose': torch.zeros((1, 48)),
                        'shape': shape,
                        'trans': torch.zeros((1, 3)),
                        'hand_type': hand_type}
    mano_pose_flat = [mano_pose_right]
    mano_out = get_mano_out(mano_pose_flat, mano_layer_fitting, bWeakPersp=False, bMatPose=False)
    _, joints_flat = mano_out['verts'], mano_out['joints'] #(N, 42, 3)


    # assemble the input pose
    mano_pose_right = {'pose': torch.cat((root_pose, hand_pose), dim=1),
                        'shape': shape,
                        'trans': trans,
                        'hand_type': hand_type}
    mano_pose_articulated = [mano_pose_right]
    mano_out = get_mano_out(mano_pose_articulated, mano_layer_fitting, bWeakPersp=False, bMatPose=False)
    _, joints_articulated = mano_out['verts'], mano_out['joints'] #(N, 42, 3)
    
    
    #add pose noise
    torch.manual_seed(noise_seed)
    
    #add noise to all the joints
    joints_articulated_noisy = joints_articulated + torch.randn(joints_articulated.shape)*noise_std
    #NOTE: even though we are adding a noise of just 1 mm, we fit a MANO model on it, so the resultant joints can have a higher error
    # print("avg joint error (when adding noise): ", torch.mean(torch.norm(joints_articulated_noisy - joints_articulated, dim=2)))
    
    # fit MANO to the noisy joints
    pose_params = get_mano_pose(joints_flat, joints_articulated_noisy)
    hand_pose_new = pose_params[0][:,3:]
    root_pose_new = pose_params[0][:,:3]

    #subtract the pose_mean
    if mano_layer_ori[hand_type].flat_hand_mean == False:
        hand_pose_new = hand_pose_new - mano_layer_ori[hand_type].pose_mean[3:]
        
        
    return root_pose_new, hand_pose_new


