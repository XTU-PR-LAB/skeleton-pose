# -*- coding: utf-8 -*-
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

# import h5py
import numpy as np
# import minpy.numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
# import data_utils
# import viz

def project_skeleton_to_image( P, f, c):
  """
    Args
      P: Nx17x3 points in world coordinates
      f: Nx2x1 Camera focal length
      c: Nx2x1 Camera center

    Returns
      Proj: Nx17x2 points in pixel space
    """
  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 3
  assert P.shape[2] == 3
  X = P.transpose((0, 2, 1))  # Nx3x17
  XX = X[:, :2, :] / np.reshape(X[:, 2, :], [-1, 1, X.shape[2]])  # Nx2x17,
  Proj = f * XX + c
  Proj = Proj.transpose((0, 2, 1))   # Nx17x2
  return Proj

def project_point_to_image( P, f, c):
  """

    Args
      P: Nx3 points in world coordinates
      f: 2x1 Camera focal length
      c: 2x1 Camera center

    Returns
      Proj: Nx2 points in pixel space
    """
  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3
  X = P.T  # 3xN
  XX = X[:2, :] / X[2, :]  # 2xN
  Proj = (f * XX) + c
  Proj = Proj.T
  return Proj

def back_project_skeleton_to_world(P_prediction, P_image, f, c, L1_square):
  P_recovered = np.zeros_like(P_prediction)
  z1_original = P_prediction[:, 0, 2]
  z1_original = np.reshape(z1_original, [-1, 1])  # Nx1
  z1 = copy.deepcopy(z1_original)

  hip_image = P_image[:, 0, :]
  X = hip_image - c  # Nx2

  XX = z1 * X  # Nx2
  xy = XX / f
  xyz = np.concatenate((xy, z1), axis=1)  # Nx3
  P_recovered[:, 0, :] = xyz

  for i in range(3):
    p1 = np.reshape(P_image[:, i, :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, i + 1, :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, i, 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, i+1, 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  #Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 1, :] = p_recovered
  z1 = copy.deepcopy(z1_original)
  lower_joint = [0, 4, 5]
  upper_joint = [4, 5, 6]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 3 + i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 4, :] = p_recovered

  z1 = copy.deepcopy(z1_original)
  lower_joint = [0, 7]
  upper_joint = [7, 8]

  for i in range(2):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 6 + i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 7, :] = p_recovered

  z1_original = copy.deepcopy(z1)

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 9]
  upper_joint = [9, 10]

  for i in range(2):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 8 + i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 9, :] = p_recovered

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 11, 12]
  upper_joint = [11, 12, 13]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 10 + i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 11, :] = p_recovered

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 14, 15]
  upper_joint = [14, 15, 16]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 13 + i], [-1, 1])  # Nx1
    p_recovered = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 14, :] = p_recovered

  return P_recovered

def eval_back_project_skeleton_to_world(z1, P_prediction, P_image, f, c, L1_square):
  B = np.shape(P_prediction)[0]
  valid_flag = np.ones([B, 1], dtype=np.float64)

  P_recovered = np.zeros_like(P_prediction)
  z1_original = z1  # P_prediction[:, 0, 2]
  z1_original = np.reshape(z1_original, [-1, 1])
  z1 = copy.deepcopy(z1_original)

  hip_image = P_image[:, 0, :]
  X = hip_image - c  # Nx2

  XX = z1 * X  # Nx2
  xy = XX / f
  xyz = np.concatenate((xy, z1), axis=1)  # Nx3
  P_recovered[:, 0, :] = xyz

  for i in range(3):
    p1 = np.reshape(P_image[:, i, :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, i + 1, :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, i, 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, i+1, 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  #Nx3
    valid_flag = valid_flag * delta_flag
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 1, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1 = copy.deepcopy(z1_original)
  lower_joint = [0, 4, 5]
  upper_joint = [4, 5, 6]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 3 + i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 4, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1 = copy.deepcopy(z1_original)
  lower_joint = [0, 7]
  upper_joint = [7, 8]

  for i in range(2):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 6 + i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 7, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1_original = copy.deepcopy(z1)

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 9]
  upper_joint = [9, 10]

  for i in range(2):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 8 + i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 9, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 11, 12]
  upper_joint = [11, 12, 13]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 10 + i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 11, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1 = copy.deepcopy(z1_original)
  lower_joint = [8, 14, 15]
  upper_joint = [14, 15, 16]

  for i in range(3):
    p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = np.reshape(L1_square[:, 13 + i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    P_recovered[:, i + 14, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  return P_recovered, valid_flag


def check_solve(delta_z, p1, p2, f, z1, L1_square):
  """

  :param delta_z: 1xN
  :param p1: 2xN
  :param p2: 2xN
  :param f: 2xN
  :param z1: 2xN
  :param L1_square: 1xN
  :return:
  """
  L1_recovered_square = delta_z**2 + (p1[0, :] * z1 / f[0, :] - p2[0, :] * (z1 + delta_z) / f[0, :])**2 + \
                        (p1[1, :] * z1 / f[1, :] - p2[1, :] * (z1 + delta_z) / f[1, :])**2


def get_best_recovered_pose(recovered_pose, recovered_z_diff, recovered_valid, joint_size, start_joint_no):
  '''

  :param recovered_pose:
  :param recovered_z_diff:
  :param recovered_valid:
  :param joint_size:
  :param start_joint_no:
  :return: best_recovered_pose, [17, 3], best_recovered_valid, [1]
  '''
  best_recovered_pose = copy.deepcopy(recovered_pose)
  best_recovered_valid = copy.deepcopy(recovered_valid)
  sorted_recovered_z_diff = np.sort(recovered_z_diff, axis=1)
  for i in range(start_joint_no, joint_size, 1):
    temp_relative_z = sorted_recovered_z_diff[:, :i]  # [n, 5])
    temp_relative_z = np.sum(temp_relative_z, axis=1)  # [n])
    min_value = np.min(temp_relative_z)

    min_idx = np.where(np.fabs(temp_relative_z - min_value) < 1.0e-8)

    min_z_diff_sum = temp_relative_z[min_idx[0]]  # 9/10 added
    sorted_recovered_z_diff = sorted_recovered_z_diff[min_idx[0]]
    best_recovered_pose = best_recovered_pose[min_idx[0]]
    best_recovered_valid = best_recovered_valid[min_idx[0]]
    min_z_diff_mean = min_z_diff_sum / float(i)  # 9/10 added
    if len(min_idx[0]) < 2:
        break

  if best_recovered_pose.shape[0] > 1:
    best_recovered_pose = best_recovered_pose[0]
    best_recovered_valid = best_recovered_valid[0]
    min_z_diff_mean = min_z_diff_mean[0]

  best_recovered_pose = np.reshape(best_recovered_pose, [-1,3])
  best_recovered_valid = np.reshape(best_recovered_valid, [-1])

  min_z_diff_mean = np.reshape(min_z_diff_mean, [-1])

  return best_recovered_pose, best_recovered_valid, min_z_diff_mean



def get_back_project_skeleton_to_world(z_root_prediction, z_relative_prediction, P_image, f, c, L1_square, root_idx, skeleton_tree, segment_no_tree):

  batch_size = np.shape(P_image)[0]
  joint_size = np.shape(P_image)[1]
  P_recovered = np.zeros([batch_size, joint_size, 3], dtype=np.float64)
  z1_original = z_root_prediction  # Nx1
  z1 = copy.deepcopy(z1_original)

  hip_image = P_image[:, root_idx, :]
  X = hip_image - c  # Nx2

  XX = z1 * X  # Nx2
  xy = XX / f
  xyz = np.concatenate((xy, z1), axis=1)  # Nx3
  P_recovered[:, root_idx, :] = xyz

  valid_flag = np.ones([batch_size, 1], dtype=np.float64)
  recovered_pose, recovered_valid = iter_back_project_skeleton_to_world(P_recovered, P_image, f, c, L1_square, skeleton_tree, root_idx, [root_idx], valid_flag, segment_no_tree)

  joint_idx = [i for i in range(17)]  # [0, 1, 2, ..., 16]
  not_root_idx = list(set(joint_idx).difference(set([root_idx])))

  z_except_root = recovered_pose[:, :, not_root_idx, 2]  # [N, number, 16]
  z_root = recovered_pose[:, :, root_idx, 2]  # [N, number]
  z_root = np.expand_dims(z_root, axis=2)  # [N, number, 1]
  recovered_relative_pose = z_except_root - z_root  # [N, number, 16]

  P_prediction_relative = z_relative_prediction # [N, 16]
  P_prediction_relative = np.expand_dims(P_prediction_relative, axis=1)  # [N, 1, 16]

  z_differ = recovered_relative_pose - P_prediction_relative   # [N, number, 16]
  z_differ = np.square(z_differ)  # [N, number, 16]
  recovered_valid_flag = True
  if recovered_valid_flag:
    temp_recovered_valid = copy.deepcopy(recovered_valid)
    temp_valid_sum = np.sum(temp_recovered_valid, axis=1)  # [N]
    temp_valid_sum = np.expand_dims(temp_valid_sum, axis=1)  # [N, 1]
    temp_valid_sum = np.repeat(temp_valid_sum, recovered_valid.shape[1], axis=1)  # [N, number]
    temp_valid_ones = np.ones_like(recovered_valid)
    temp_recovered_valid = np.where(temp_valid_sum < 0.1, temp_valid_ones, temp_recovered_valid)
    temp_recovered_valid = np.reshape(temp_recovered_valid, [batch_size, -1])
    temp_recovered_valid = np.expand_dims(temp_recovered_valid, axis=2)
    temp_recovered_valid = np.repeat(temp_recovered_valid, joint_size - 1, axis=2)  # [N, number, 16]
    biggest_threshold = np.full([z_differ.shape[0], z_differ.shape[1], z_differ.shape[2]], 1e10)
    z_differ = np.where(temp_recovered_valid > 0.1, z_differ, biggest_threshold)
  best_recovered_pose = np.zeros([batch_size, joint_size, 3], dtype=np.float64)
  best_recovered_valid = np.zeros([batch_size], dtype=np.float64)
  best_recovered_z_diff_mean = np.zeros([batch_size], dtype=np.float64)
  for i in range(batch_size):
    temp_recovered_pose = recovered_pose[i]  # [number, 17, 3]
    temp_z_differ = z_differ[i]  # [number, 16]
    temp_recovered_valid = recovered_valid[i]  # [number]
    best_recovered_pose[i, :, :], best_recovered_valid[i], best_recovered_z_diff_mean[i] = get_best_recovered_pose(temp_recovered_pose, temp_z_differ, temp_recovered_valid, joint_size, 16)

  return best_recovered_pose,best_recovered_valid, best_recovered_z_diff_mean  # [N, 17, 3], [N]


def test_back_project_skeleton_to_world(P_prediction, P_image, f, c, L1_square, root_idx, skeleton_tree, segment_no_tree):

  recovered_valid_flag = True
  P_recovered = np.zeros_like(P_prediction)
  z1_original = P_prediction[:, root_idx, 2]
  z1_original = np.reshape(z1_original, [-1, 1])  # Nx1
  z1 = copy.deepcopy(z1_original)

  hip_image = P_image[:, root_idx, :]
  X = hip_image - c  # Nx2

  XX = z1 * X  # Nx2
  xy = XX / f
  xyz = np.concatenate((xy, z1), axis=1)  # Nx3
  P_recovered[:, root_idx, :] = xyz
  batch_size = np.shape(P_image)[0]
  valid_flag = np.ones([batch_size, 1], dtype=np.float64)
  recovered_pose, recovered_valid = iter_back_project_skeleton_to_world(P_recovered, P_image, f, c, L1_square, skeleton_tree, root_idx, [root_idx], valid_flag, segment_no_tree)

  joint_idx = [i for i in range(17)]  # [0, 1, 2, ..., 16]
  not_root_idx = list(set(joint_idx).difference(set([root_idx])))

  z_except_root = recovered_pose[:, :, not_root_idx, 2]  # [N, number, 16]
  z_root = recovered_pose[:, :, root_idx, 2]  # [N, number]
  z_root = np.expand_dims(z_root, axis=2)  # [N, number, 1]
  recovered_relative_pose = z_except_root - z_root  # [N, number, 16]

  P_prediction_except_root = P_prediction[:, not_root_idx, 2]  # [N, 16]
  P_prediction_root = P_prediction[:, root_idx, 2]  # [N]
  P_prediction_root = np.expand_dims(P_prediction_root, axis=1)  # [N, 1]
  P_prediction_relative = P_prediction_except_root - P_prediction_root  # [N, 16]
  P_prediction_relative = np.expand_dims(P_prediction_relative, axis=1)  # [N, 1, 16]

  z_differ = recovered_relative_pose - P_prediction_relative   # [N, number, 16]
  z_differ = np.square(z_differ)  # [N, number, 16]
  z_differ = np.sum(z_differ, axis=2)  # [N, number]

  if recovered_valid_flag:
    recovered_valid = np.reshape(recovered_valid, [recovered_valid.shape[0], -1])  # [N, number]
    biggest_threshold = np.full([z_differ.shape[0], z_differ.shape[1]], 1e10)
    z_differ = np.where(recovered_valid > 0.1, z_differ, biggest_threshold)

  best_idx = np.argmin(z_differ, axis=1)

  batch_idx = np.arange(batch_size)
  best_recovered_pose = recovered_pose[batch_idx, best_idx]
  return best_recovered_pose  # [N, 17, 3]

def iter_back_project_skeleton_to_world(recovered_P_input, P_image, f, c, L1_square, skeleton_tree, father_idx, ascend_idx, validate_flag, segment_no_tree):

  if isinstance(skeleton_tree[0],list):  # list, fork
    joint_idx = [i for i in range(17)]  # [0, 1, 2, ..., 16]
    child_idx = list(set(joint_idx).difference(set(ascend_idx)))
    for i, item in enumerate(skeleton_tree):
      child_recovered_pose, child_recovered_flag = iter_back_project_skeleton_to_world(recovered_P_input, P_image, f, c, L1_square,
                                                                                       skeleton_tree[i], father_idx, ascend_idx, validate_flag, segment_no_tree[i])

      if i == 0:
        recovered_pose = child_recovered_pose
        recovered_flag = child_recovered_flag
      else:
        batch_size = np.shape(recovered_pose)[0]
        number = np.shape(recovered_pose)[1]
        child_number = np.shape(child_recovered_pose)[1]
        recovered_pose = np.expand_dims(recovered_pose, axis=2)  # [b, number, 1, 17, 3]
        recovered_pose = np.repeat(recovered_pose, child_number, axis=2) # [b, number, child_number, 17, 3]
        child_recovered_pose = np.expand_dims(child_recovered_pose, axis=1) # [b, 1, child_number, 17, 3]
        child_recovered_pose = np.repeat(child_recovered_pose, number, axis=1)  # [b, number, child_number, 17, 3]
        recovered_pose[:, :, :, child_idx, :] = recovered_pose[:, :, :, child_idx, :] + child_recovered_pose[:, :, :, child_idx, :]
        recovered_pose = np.reshape(recovered_pose, [batch_size, -1, 17, 3])  # [b, n, 17, 3]

        recovered_flag = np.expand_dims(recovered_flag, axis=2)  # [b, number, 1]
        recovered_flag = np.repeat(recovered_flag, child_number, axis=2)  # [b, number, child_number, 1]
        child_recovered_flag = np.expand_dims(child_recovered_flag, axis=1)  # [b, 1, child_number, 1]
        child_recovered_flag = np.repeat(child_recovered_flag, number, axis=1)  # [b, number, child_number, 1]
        recovered_flag = recovered_flag * child_recovered_flag # [b, number, child_number, 1]
        recovered_flag = np.reshape(recovered_flag, [batch_size, -1, 1])  # [b, n, 1]

    return recovered_pose, recovered_flag
  else:
    recovered_pose0,recovered_pose1, recovered_flag0, recovered_flag1= iter_back_project_point_to_world(P_image, f, c, recovered_P_input, L1_square, father_idx, skeleton_tree[0], validate_flag, segment_no_tree[0])
    if(len(skeleton_tree) == 1): # leaf node
      recovered_pose0 = np.expand_dims(recovered_pose0, axis=1)  # [b, 1, 17, 3]
      recovered_pose1 = np.expand_dims(recovered_pose1, axis=1)  # [b, 1, 17, 3]
      recovered_flag0 = np.expand_dims(recovered_flag0, axis=1)  # [b, 1, 1]
      recovered_flag1 = np.expand_dims(recovered_flag1, axis=1)  # [b, 1, 1]
      recovered_pose = np.concatenate((recovered_pose0, recovered_pose1),axis=1) # [b, 2, 17, 3]
      recovered_flag = np.concatenate((recovered_flag0, recovered_flag1),axis=1) # [b, 2, 1]
      return recovered_pose,recovered_flag
    else:
      temp_ascent_idx = ascend_idx + [skeleton_tree[0]]
      temp_recovered_flag0 = validate_flag * recovered_flag0
      temp_recovered_flag1 = validate_flag * recovered_flag1
      recovered_pose_0, recovered_flag_0 = iter_back_project_skeleton_to_world(recovered_pose0, P_image, f, c,  L1_square, skeleton_tree[1], skeleton_tree[0], temp_ascent_idx, temp_recovered_flag0, segment_no_tree[1])
      recovered_pose_1, recovered_flag_1 = iter_back_project_skeleton_to_world(recovered_pose1, P_image, f, c,  L1_square,
                                                                            skeleton_tree[1], skeleton_tree[0], temp_ascent_idx,
                                                                            temp_recovered_flag1, segment_no_tree[1])
      return np.concatenate((recovered_pose_0, recovered_pose_1),axis=1), np.concatenate((recovered_flag_0, recovered_flag_1),axis=1)  # [b, n, 17, 3], [b, n, 1]

def iter_back_project_point_to_world(p_image, f_input, cc_input, recovered_P_input, L1_square_input, father_idx, this_idx, validate_flag, seg_no):

  p1 = p_image[:, father_idx, :]
  p1 = p1.T  # 2xN
  p2 = p_image[:, this_idx, :]
  p2 = p2.T  # 2xN

  cc = cc_input.T  # 2xN
  p1 = p1 - cc
  p2 = p2 - cc
  f = f_input.T  # 2xN
  z1_input = recovered_P_input[:, father_idx, 2]
  z1 = z1_input.T  # 1xN

  L1_square = L1_square_input[:, seg_no]  # Nx1
  L1_square = L1_square.T  # 1xN
  a = (f[0, :] * f[1, :])**2 + (p2[0, :] * f[1, :])**2 + (p2[1, :] * f[0, :])**2  # Nx1
  a = np.reshape(a, [1, -1])  # 1xN
  b = -2.0 * p1[0, :] * z1 * p2[0, :] * ((f[1, :])**2) + 2.0 * ((p2[0, :])**2) * z1 * (f[1, :])**2 - 2.0 * p1[1, :] \
      * z1 * p2[1, :] * ((f[0, :])**2) + 2.0 * ((p2[1, :])**2) * z1 * ((f[0, :])**2)
  c = (p1[0, :] * z1 * f[1, :])**2 - 2.0 * p1[0, :] * p2[0, :] * (z1 * f[1, :])**2 + (p2[0, :] * z1 * f[1, :])**2 \
      + (p1[1, :] * z1 * f[0, :])**2 - 2.0 * p1[1, :] * p2[1, :] * (z1 * f[0, :])**2 + (p2[1, :] * z1 * f[0, :])**2 \
      - L1_square * (f[0, :] * f[1, :])**2

  min_threshold = np.full([a.shape[0], a.shape[1]], 1e-8)
  b_ac = b**2 - 4.0 * a * c
  numer1 = np.where(b_ac > np.zeros_like(a, dtype=np.float64), (-b + np.sqrt(b_ac)),
  np.zeros_like(a, dtype=np.float64))
  numer2 = np.where(b_ac > np.zeros_like(a, dtype=np.float64),
                    (-b - np.sqrt(b_ac)),
                    np.zeros_like(a, dtype=np.float64))

  delta_z1 = np.where(np.abs(a) < min_threshold, np.zeros_like(a, dtype=np.float64), numer1 / (2.0 * a))
  delta_z2 = np.where(np.abs(a) < min_threshold, np.zeros_like(a, dtype=np.float64), numer2 / (2.0 * a))


  z2_delta_z1 = z1 + delta_z1  # 1xN
  x2_y2_delta_z1 = z2_delta_z1 * p2 / f  # 2xN
  recovered_delta_z1 = np.concatenate((x2_y2_delta_z1, z2_delta_z1),axis=0)  # 3xN
  recovered_delta_z1 = recovered_delta_z1.T

  delta_flag = np.where(np.abs(a) < min_threshold, np.zeros_like(a, dtype=np.float64),
                        np.ones_like(a, dtype=np.float64))
  delta_flag2 = np.where(b_ac > np.zeros_like(a, dtype=np.float64), np.ones_like(a, dtype=np.float64),
                         np.zeros_like(a, dtype=np.float64))
  delta_flag = delta_flag * delta_flag2
  delta_flag = delta_flag.T  # [N, 1]
  delta_flag = delta_flag * validate_flag
  delta2_flag = copy.deepcopy(delta_flag)

  z2_delta_z2 = z1 + delta_z2  # 1xN
  x2_y2_delta_z2 = z2_delta_z2 * p2 / f  # 2xN
  recovered_delta_z2 = np.concatenate((x2_y2_delta_z2, z2_delta_z2), axis=0)  # 3xN
  recovered_delta_z2 = recovered_delta_z2.T  # Nx3

  recovered_P_delta_z1 = copy.deepcopy(recovered_P_input)
  recovered_P_delta_z1[:, this_idx, :] = recovered_delta_z1

  recovered_P_delta_z2 = copy.deepcopy(recovered_P_input)
  recovered_P_delta_z2[:, this_idx, :] = recovered_delta_z2

  return recovered_P_delta_z1, recovered_P_delta_z2, delta_flag, delta2_flag  # [N, 17, 3], [N, 1]

def back_project_point_to_world(p1_input, p2_input, f_input, cc_input, z1_input, z1_predict_input, z2_predict_input, L1_square_input):

  p1 = p1_input.T  # 2xN
  p2 = p2_input.T  # 2xN
  cc = cc_input.T  # 2xN
  p1 = p1 - cc
  p2 = p2 - cc
  f = f_input.T  # 2xN
  z1 = z1_input.T  # 1xN
  z1_predict = z1_predict_input.T  # 1xN
  z2_predict = z2_predict_input.T  # 1xN
  L1_square = L1_square_input.T  # 1xN
  a = (f[0, :] * f[1, :])**2 + (p2[0, :] * f[1, :])**2 + (p2[1, :] * f[0, :])**2  # Nx1
  a = np.reshape(a, [1, -1])  # 1xN
  b = -2.0 * p1[0, :] * z1 * p2[0, :] * ((f[1, :])**2) + 2.0 * ((p2[0, :])**2) * z1 * (f[1, :])**2 - 2.0 * p1[1, :] \
      * z1 * p2[1, :] * ((f[0, :])**2) + 2.0 * ((p2[1, :])**2) * z1 * ((f[0, :])**2)
  c = (p1[0, :] * z1 * f[1, :])**2 - 2.0 * p1[0, :] * p2[0, :] * (z1 * f[1, :])**2 + (p2[0, :] * z1 * f[1, :])**2 \
      + (p1[1, :] * z1 * f[0, :])**2 - 2.0 * p1[1, :] * p2[1, :] * (z1 * f[0, :])**2 + (p2[1, :] * z1 * f[0, :])**2 \
      - L1_square * (f[0, :] * f[1, :])**2

  min_threshold = np.full([a.shape[0], a.shape[1]], 1e-8)
  b_ac = b**2 - 4.0 * a * c
  numer1 = np.where(b_ac > np.zeros_like(a, dtype=np.float64), (-b + np.sqrt(b_ac)),
  np.zeros_like(a, dtype=np.float64))
  numer2 = np.where(b_ac > np.zeros_like(a, dtype=np.float64),
                    (-b - np.sqrt(b_ac)),
                    np.zeros_like(a, dtype=np.float64))

  delta_z1 = np.where(np.abs(a) < min_threshold, np.zeros_like(a, dtype=np.float64), numer1 / (2.0 * a))
  delta_z2 = np.where(np.abs(a) < min_threshold, np.zeros_like(a, dtype=np.float64), numer2 / (2.0 * a))


  delta_z_predict = z2_predict - z1_predict
  delta_z = np.where(delta_z1 * delta_z2 < 0, np.where(z2_predict > z1_predict, delta_z1, delta_z2), np.where(np.abs(delta_z1 - delta_z_predict) < np.abs(delta_z2 - delta_z_predict), delta_z1, delta_z2))
  z2 = z1 + delta_z  # 1xN
  x2_y2 = z2 * p2 / f  # 2xN
  recovered = np.concatenate((x2_y2, z2),axis=0)  # 3xN
  recovered = recovered.T

  delta_flag = np.where(np.abs(a) < min_threshold, np.zeros_like(delta_z, dtype=np.float64),
                        np.ones_like(delta_z, dtype=np.float64))
  delta_flag2 = np.where(b_ac > np.zeros_like(a, dtype=np.float64), np.ones_like(a, dtype=np.float64),
                         np.zeros_like(a, dtype=np.float64))
  delta_flag = delta_flag * delta_flag2
  delta_flag = delta_flag.T  # [N, 1]

  return recovered, delta_flag


def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) ) # 1 + k1r^2 + k2r^4 + k3r^6
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2

def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate

  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T




