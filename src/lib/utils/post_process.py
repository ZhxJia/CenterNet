from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot
from .image import get_affine_transform, affine_transform


def get_pred_depth(depth, ref_depth=None):
    '''
    depth = mu_z + sigma_z * delta_z
    '''
    if ref_depth is not None:
        depth = depth * ref_depth[1] + ref_depth[0]
        return depth
    else:
        return depth


def get_alpha_8(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def get_alpha_2(rot):
    '''
    Input:
        rot: (B, 2) [sin_a, cos_a]
    Return:
        rot[B, ]
    '''
    alphas = np.arctan2(rot[:, 0], rot[:, 1])
    cos_pos_idx = (rot[:, 1] >= 0).nonzero()
    cos_neg_idx = (rot[:, 1] < 0).nonzero()

    alphas[cos_pos_idx] -= np.pi / 2
    alphas[cos_neg_idx] += np.pi / 2

    return alphas


def get_dimension(cls_id, dims_offset, ref_dim):
    ref_dim = ref_dim[cls_id, :]
    dimensions = np.exp(dims_offset) * np.diag(ref_dim)

    return dimensions


def ddd_post_process_2d(dets, c, s, opt):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    include_wh = dets.shape[2] > 16
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
        classes = dets[i, :, -1]
        for j in range(opt.num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :3].astype(np.float32),
                get_alpha_8(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
                get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
                dets[i, inds, 12:15].astype(np.float32)], axis=1)
            if include_wh:
                top_preds[j + 1] = np.concatenate([
                    top_preds[j + 1],
                    transform_preds(
                        dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
                        .astype(np.float32)], axis=1)
        ret.append(top_preds)
    return ret


def ddd_post_process_3d(dets, calibs):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    for i in range(len(dets)):
        preds = {}
        for cls_ind in dets[i].keys():
            preds[cls_ind] = []
            for j in range(len(dets[i][cls_ind])):
                center = dets[i][cls_ind][j][:2]
                score = dets[i][cls_ind][j][2]
                alpha = dets[i][cls_ind][j][3]
                depth = dets[i][cls_ind][j][4]
                dimensions = dets[i][cls_ind][j][5:8]
                wh = dets[i][cls_ind][j][8:10]
                locations, rotation_y = ddd2locrot(
                    center, alpha, dimensions, depth, calibs[0])
                bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                        center[0] + wh[0] / 2, center[1] + wh[1] / 2]
                pred = [alpha] + bbox + dimensions.tolist() + \
                       locations.tolist() + [rotation_y, score]
                preds[cls_ind].append(pred)
            preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
        ret.append(preds)
    return ret


def ddd_post_process(dets, c, s, calibs, opt):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    dets = ddd_post_process_2d(dets, c, s, opt)
    dets = ddd_post_process_3d(dets, calibs)
    return dets


def smk3d_generate3d(dets, calibs):
    '''
    Input: dets class-based list, shape: batch,dict{cls_id:objects_attribute:[x,y,scores,alpha,depth,dimension]}
           calib: matrix 3*4
    Return: corners_3d: 8x3
    '''
    ret = []
    for i in range(len(dets)):
        preds = {}
        for cls_ind in dets[i].keys():
            preds[cls_ind] = []
            for j in range(len(dets[i][cls_ind])):
                proj_ct = dets[i][cls_ind][j][:2]
                score = dets[i][cls_ind][j][2]     #8
                alpha = dets[i][cls_ind][j][3]     #0
                depth = dets[i][cls_ind][j][4]
                dimension = dets[i][cls_ind][j][5:8] #1,2,3
                location, rotation_y = ddd2locrot(
                    proj_ct, alpha, dimension, depth, calibs[0])  #4,5,6,7 todo debug : calibs
                pred = [alpha] + dimension.tolist() + location.tolist() + [rotation_y, score]
                preds[cls_ind].append(pred)
            preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
        ret.append(preds)
    return ret


def smk3d_post_process(dets, c, s, calibs, opt):
    '''
    based on the origin network output, retrieve the 3D objects
    Input:
      dets: [xs:0, ys:1, scores:2, rot:3,4, depth_offset:5, dim_offset:6,7,8, clses:9] , shape: batch,max_dets,dim
    '''
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
        classes = dets[i, :, -1]
        for j in range(opt.num_calsses):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :3].astype(np.float32),
                get_alpha_2(dets[i, inds, 3:5])[:, np.newaxis].astype(np.float32),
                get_pred_depth(dets[i, inds, 5:6], ref_depth=opt.reference_depth).astype(np.float32),
                get_dimension(j, dets[i, inds, 6:9], opt.reference_size).astype(np.float32)
            ], axis=1)
        ret.append(top_preds)

    ret_3d = smk3d_generate3d(ret, calibs)

    return ret_3d


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim, dim: bboxes:4 scores:1, clses:1
    # h,w -> output_w, output_h (128,128)
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret
