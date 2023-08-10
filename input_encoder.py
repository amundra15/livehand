import numpy as np
import torch
import torch.nn.functional as F
from fairnr.clib._ext import point_face_dist_forward, point_face_dist_backward
from torch.autograd import Function
from torch.autograd.function import once_differentiable


EPSILON = 1e-8


def read_mano_uv_obj(filename):
    vt, ft, f = [], [], []
    for content in open(filename):
        if content.startswith('#'):
            continue
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
            f.append([int(a.split('/')[0]) for a in contents[1:] if a])
    
    #NOTE: the obj file is 1-indexed, thus we need to minus 1
    vt, ft, f = np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1, np.array(f, dtype=np.long) - 1     #Neural Actor encoder.py line 1241
    
    #invert the v coordinate
    vt[:, 1] = 1 - vt[:, 1]
    
    return vt, ft, f


def save_obj_for_debugging(xyz, r, g, b, filename):
    # #save in an obj file for debugging with entries: V x y z r g b
    # with open(filename, 'a+') as f:
    with open(filename, 'w') as f:
        for i in range(xyz.shape[0]):
            f.write('v %f %f %f %f %f %f \n' % (xyz[i,0], xyz[i,1], xyz[i,2], r[i], g[i], b[i]))


def get_uvd(pts, mesh_vertices, mesh_faces, mesh_face_uv):
    '''
    get the uv coordinates based on the nearest mesh face
    pts: sampling points in MANO reference frame
    '''
    
    triangles = F.embedding(mesh_faces, mesh_vertices)       #inputs: indices, lookup_table          #torch.Size([1538, 3, 3])
    l_idx = torch.tensor([0,]).type_as(mesh_faces)
    # min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(pts, l_idx, triangles, l_idx, pts.size(0))
    min_dis, min_face_idx, w0, w1, w2 = point_face_dist(pts, l_idx, triangles, l_idx, pts.size(0))
    bary_coords = torch.stack([w0, w1, w2], 1)   # B x 3
    sampled_uvs = (mesh_face_uv[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)

    #min_dis is actually distance^2. convert it to signed distance
    A, B, C = triangles[:,0], triangles[:,1], triangles[:,2]
    triangle_normals = torch.cross((B - A), (C - B), 1)
    faces_xyzs = (triangles[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)
    insideout = ((pts - faces_xyzs) * triangle_normals[min_face_idx]).sum(-1).sign()
    signed_dist = ((min_dis+EPSILON).sqrt() * insideout)[..., None]
    
    intermediates = {'min_face_idx': min_face_idx, 'bary_coords': bary_coords}
    
    # save_obj_for_debugging(xyz=pts.cpu().numpy(), r=(insideout==1), g=(min_dis/min_dis.max()).sqrt().cpu().numpy(), b=(insideout==-1), filename='vis/sampling_points.obj')
    

    return sampled_uvs, signed_dist, intermediates



#write a custom Function to find the nearest face in a differentiable way
class PointFaceDistFunction(Function):
    
    @staticmethod
    def forward(ctx, pts, l_idx, triangles, l_idx2, num_samples):
        '''
        pts: torch.Size([num_samples, 3])
        triangles: torch.Size([num_faces, 3, 3])
        '''
        
        min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(pts, l_idx, triangles, l_idx2, num_samples)
        
        ctx.save_for_backward(pts, triangles, min_face_idx)
        
        return min_dis, min_face_idx, w0, w1, w2
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_min_dis, grad_min_face_idx, grad_w0, grad_w1, grad_w2):
        
        pts, triangles, min_face_idx = ctx.saved_tensors
        
        grad_pts, grad_triangles = point_face_dist_backward(pts, triangles, min_face_idx, grad_min_dis)
        
        # return grad_pts, None, grad_triangles, None, None         #NOTE: Returning gradients for inputs that don't require it is not an error. -PyTorch documentation
        return grad_pts, None, None, None, None
    
point_face_dist = PointFaceDistFunction.apply