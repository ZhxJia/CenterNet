from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat

PI = 3.14159265


class SMOKECoder():
    def __init__(self, depth_ref, dim_ref, device="cuda"):
        self.depth_ref = torch.as_tensor(depth_ref, dtype=torch.float32).to(device=device)
        self.dim_ref = torch.as_tensor(dim_ref, dtype=torch.float32).to(device=device)

    def decode_depth(self, depths_offset):
        '''
        Transform depth offset to depth
        '''
        self.depth_ref = self.depth_ref.to(depths_offset)
        depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        return depth

    def decode_location(self, points, points_offset, depths, dims, Ks, trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y) [N, 50, 2]
            points_offset: project points offset in (delata_x, delta_y) [N, 50, 2]
            depths: object depth z [N, 50, 1]
            Ks: camera intrinsic matrix, shape = [N, 3, 4]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N,50,3]
        '''
        N, M = points.shape[0], points.shape[1]  # batch size and max_objects number
        proj_points = points + points_offset
        # transform points to homogeneous coord
        homo_points = torch.cat((proj_points, torch.ones(N, M, 1, device=points.device)), dim=2)
        # transform projected points to raw image
        trans_mats_inv = trans_mats.inverse()  # N,3,3
        trans_mats_inv_t = trans_mats_inv.transpose(1, 2)
        img_points = torch.matmul(homo_points, trans_mats_inv_t)  # [5, 50, 3]
        locations = self.unproject_2d_to_3d(img_points, depths, Ks)
        locations[:, :, 1] += dims[:, :, 0] / 2  # located at the bottom of the object
        return locations

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id , shape= [N, 50]
            dims_offset: dimension offsets, shape = (N, 50 ,3)

        Returns:
            decode dimension: [N, 50, 3]
        '''
        cls_flat = cls_id.flatten().long()
        ref_dim = self.dim_ref[cls_flat, :].reshape(dims_offset.shape[0], dims_offset.shape[1], -1)
        dimensions = torch.exp(dims_offset) * ref_dim.to(dims_offset.device)

        return dimensions

    def decode_orientation(self, ori_vector, locations):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format, shape:[N,50,2]
            locations: object location, shape:[N,50,3]

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''
        ori_vector_flatten = ori_vector.view(-1, ori_vector.shape[-1])
        locations_flatten = locations.view(-1, locations.shape[-1])

        theta_rays = torch.atan(locations_flatten[:, 0] / (locations_flatten[:, 2] + 1e-7))
        alphas = torch.atan(ori_vector_flatten[:, 0] / (ori_vector_flatten[:, 1] + 1e-7))  # [N,50]

        # get cosine value positive and negative index
        cos_pos_idx = (ori_vector_flatten[:, 1] >= 0).nonzero()
        cos_neg_idx = (ori_vector_flatten[:, 1] < 0).nonzero()

        alphas[cos_pos_idx] -= np.pi / 2
        alphas[cos_neg_idx] += np.pi / 2  # Kitti provided alpha_x, we choose regress alpha_z

        # retrieve object rotation y angle
        rotys = alphas + theta_rays

        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < -np.pi).nonzero()

        if len(larger_idx != 0):
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx != 0):
            rotys[small_idx] += 2 * np.pi

        rotys = rotys.view(ori_vector.shape[0], ori_vector.shape[1], 1)
        alphas = alphas.view(ori_vector.shape[0], ori_vector.shape[1], 1)

        return rotys, alphas

    def unproject_2d_to_3d(self, pt_2ds, depths, P):
        # pts_2d: [N, 50, 3]: homogeneous format
        # depth: [N, 50, 1]
        # P: calibration [N, 3, 4]
        # return: pt3d: [N, 50, 3]
        z = depths - P[:, 2:3, 3:4]
        x = (pt_2ds[:, :, [0]] * depths - P[:, :1, 3:4] - P[:, :1, 2:3] * z) / P[:, :1, :1]
        y = (pt_2ds[:, :, [1]] * depths - P[:, 1:2, 3:4] - P[:, 1:2, 2:3] * z) / P[:, 1:2, 1:2]
        pt_3ds = torch.cat((x, y, z), dim=2)
        return pt_3ds

    @staticmethod
    def rad_to_matrix(rotys):
        '''
        Input:
            rotys: shape [N,]
        Return：
            rotation matrix: shape [N, 3, 3]
        '''
        c, s = rotys.cos(), rotys.sin()
        temp = torch.tensor([[1, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 1]], dtype=torch.float32, device=rotys.device)
        R = temp.repeat(rotys.shape[0], 1).view(rotys.shape[0], -1, 3)
        R[:, 0, 0] *= c
        R[:, 0, 2] *= s
        R[:, 2, 0] *= s
        R[:, 2, 2] *= c
        return R

    def encode_box3d(self, rotys, dims, locs):
        '''
        Construct 3d bounding box
            Input:
                rotys: rotation Y-axis, shape [N,M,1]
                dims: dimensions of objects, shape [N,M,3]
                locs: location of objects' center , shape [N,M,3]
                where N means batch size, M means max detected objects ,see dataset sample
            Output:
                3d box of objects [N,M,3,8] eight corners
        '''
        N, M = rotys.shape[0], rotys.shape[1]
        rotys = rotys.flatten()
        dims = dims.view(-1, 3)
        locs = locs.view(-1, 3)
        diag_dims = torch.diag_embed(dims)
        h, w, l = dims[:, 0], dims[:, 1], dims[:, 2]

        R = self.rad_to_matrix(rotys)  # [NxM,3,3]
        temp = torch.tensor([[0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5],
                             [0, 0, 0, 0, -1, -1, -1, -1],
                             [0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]], dtype=torch.float32, device=rotys.device)
        temp = temp.repeat(rotys.shape[0], 1).view(rotys.shape[0], -1, 8)
        corners_local = torch.matmul(diag_dims, temp)  # [NxM ,3, 8]
        corners_3d = torch.matmul(R, corners_local)
        corners_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

        corners_3d = corners_3d.view(N, M, 3, -1)
        return corners_3d
