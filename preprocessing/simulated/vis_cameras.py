import numpy as np
import json
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch

from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D 
from matplotlib.text import Annotation

import sys


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz
    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot(111, projection='3d')



def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    zs = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        zs.append(z)
    return render_poses, zs


with open('test_cameras.json') as fp:
    cameras = json.load(fp)
    
print(cameras.keys())
# print(cameras['campos'].keys())
    
#generate a list of camera positions
cam_locations = np.array(list(cameras['campos'].values()))
ax.scatter(cam_locations[:,0], cam_locations[:,1], cam_locations[:,2], c='g')
mean_camera_location = np.mean(cam_locations, axis=0)
ax.scatter(mean_camera_location[0], mean_camera_location[1], mean_camera_location[2], c='b')

#average distance from mean camera location
avg_radius = np.mean(np.linalg.norm(cam_locations - mean_camera_location, axis=1))




#plot view-at direstion (doesnt work) #from generate_sim_data.py
t = np.array(cameras['campos']['400262'], dtype=np.float32).reshape(3)
rot = np.array(cameras['camrot']['400262'], dtype=np.float32).reshape(3,3)
# # t = -np.dot(rot,t.reshape(3,1)).reshape(3) # -Rt -> t
# ax.scatter(t[0], t[1], t[2], c='r')

# #rotate by 90 degrees around z axis
# # rotate_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
# # rot = np.dot(rotate_z, rot)
angles = R.as_rotvec(R.from_matrix(rot)) 
ax.quiver(t[0], t[1], t[2], angles[0], angles[1], angles[2], length=300, normalize=True, color='g')


# #generate cameras around the mean camera location
# rots = 1
# N = 10
# zrate = 1
# focal = 1
# up = np.array([0, 1, 0])
# c2w = np.concatenate([rot, t.reshape(3,1)], 1)
# render_poses, zs = render_path_spiral(c2w, up, rads=t, focal=focal, zrate=zrate, rots=rots, N=N)
# render_poses = np.array(render_poses)
# zs = -np.array(zs)
# # pdb.set_trace()


render_poses = torch.stack([pose_spherical(theta=angle, phi=phi, z=0, radius=avg_radius, translate=0.0) for angle in np.linspace(-180,180,5) for phi in np.linspace(-90,90,5)], 0)
#translate to mean camera location
render_poses[:, :3, 3] += torch.from_numpy(mean_camera_location)

# pdb.set_trace()

ax.scatter(render_poses[:,0,3], render_poses[:,1,3], render_poses[:,2,3], c='r')
# ax.quiver(render_poses[:,0,3], render_poses[:,1,3], render_poses[:,2,3], zs[:,0], zs[:,1], zs[:,2], length=300, normalize=True, color='red')

#convert xyz to -x,y,-z
render_poses[:, 0, :3] *= -1
render_poses[:, 2, :3] *= -1

angles = R.as_rotvec(R.from_matrix(render_poses[0,:3,:3])) 
ax.quiver(render_poses[0,0,3], render_poses[0,1,3], render_poses[0,2,3], angles[0], angles[1], angles[2], length=300, normalize=True, color='red')



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
