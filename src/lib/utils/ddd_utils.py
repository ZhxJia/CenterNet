from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 0, 255)):
  '''
  Input:
    corners: the box3d corners projected to image plane
    c: color
  '''
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
               (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
               (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def draw_box_2d(image, corners, c=(0, 0, 255)):
    '''
    Input:
        corners: the box2d corners (x1, y1, x2, y2)
    '''
    image = image.copy()
    cv2.rectangle(image, (corners[0],corners[1]), (corners[2],corners[3]), c, 2)
    return image

def draw_points(image, points, c=(0, 0, 255)):
    '''
    Input:
        points: nx2
    '''
    points = points.astype(int)
    image = image.copy()
    for i in range(points.shape[0]):
        cv2.circle(image, (points[i, 0], points[i, 1]), 3, c, -1)
    return image


def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d

def encode_label(calib, rotation_y, dim, location):
    '''
    Output:
        proj_ct: the projected center point of the local 3d center [2]
        box_2d: the 3d corners projected to the image plane [4]
        corners_3d: the 8 corners in camera coordinate 8x3
        corners_2d: the 8 corners projected to  image plane 8x2
    '''
    x, y, z = location[0], location[1], location[2]
    l, w, h = dim[2], dim[1], dim[0]

    corners_3d = compute_box_3d(dim, location, rotation_y) # 8x3
    corners_2d = project_to_image(corners_3d, calib) # 8x2

    loc_center = np.expand_dims(np.array([x, y - h / 2, z]), axis=0)
    proj_ct = project_to_image(loc_center, calib)

    box_2d = np.array([min(corners_2d[:, 0]), min(corners_2d[:, 1]),
                       max(corners_2d[:, 0]), max(corners_2d[:, 1])])
    return proj_ct, box_2d, corners_3d, corners_2d








if __name__ == '__main__':
    def read_clib(calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib
    dim = [1.57, 1.65, 3.35]
    rotation_y = -1.42
    location = [4.43, 1.65, 5.2]
    calib = read_clib("../../../data/kitti/training/calib/000010.txt")
    depth = location[2]


    proj_ct, proj_box2d, corners_3d, corners_2d = encode_label(calib, rotation_y, dim, location)

    # location = unproject_2d_to_3d(proj_ct[0], depth, calib)
    # proj_ct, proj_box2d, corners_3d, corners_2d = encode_label(calib, rotation_y, dim, location)
    box_2d = np.array([1013.39, 182.46, 1241.00, 374.00], dtype=np.float32)
    raw_img = cv2.imread('../../../data/kitti/training/image_2/000010.png')
    cv2.imshow('0001',raw_img)
    cv2.waitKey(1000)
    image = draw_box_3d(raw_img.copy(),corners_2d)
    cv2.imshow('0002',image)
    cv2.waitKey(1000)
    img2d = draw_box_2d(raw_img.copy(), proj_box2d)
    cv2.imshow('0003',img2d)
    cv2.waitKey(1000)
    imgpoint = draw_points(raw_img, proj_ct)
    cv2.imshow('0004',imgpoint)
    cv2.waitKey(0)
    alpha = -0.20
    tl = np.array([712.40, 143.00], dtype=np.float32)
    br = np.array([810.73, 307.92], dtype=np.float32)
    ct = (tl + br) / 2
    rotation_y = 0.01
    print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
    print('rotation_y', rotation_y)
