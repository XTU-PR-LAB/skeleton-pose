# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import tensorflow as tf
import copy


def back_project_point_to_world(p1_input, p2_input, f_input, cc_input, z1_input, z1_predict_input, z2_predict_input, L1_square_input):
  """

  Args
    p1_input: Nx2 start points in image coordinates，
    p2_input: Nx2 end points in image coordinates，
    f_input: Nx2 Camera focal length
    cc_input: Nx2 Camera center
    z1_input: Nx1 the z coordinate of the start point
    z1_predict_input, z2_predict: Nx1
    L1_square_input: Nx1 the square of length of the segment

  Returns
    Rec: Nx3 points in camera space
  """
  p1 = tf.transpose(p1_input, perm=[1, 0])  # p1 = p1_input.T  # 2xN
  p2 = tf.transpose(p2_input, perm=[1, 0])  # p2 = p2_input.T  # 2xN
  cc = tf.transpose(cc_input, perm=[1, 0])  # cc = cc_input.T  # 2xN
  p1 = p1 - cc
  p2 = p2 - cc
  f = tf.transpose(f_input, perm=[1, 0])  # f = f_input.T  # 2xN
  z1 = tf.transpose(z1_input, perm=[1, 0])  # z1 = z1_input.T  # 1xN
  z1_predict = tf.transpose(z1_predict_input, perm=[1, 0])  # z1_predict = z1_predict_input.T  # 1xN
  z2_predict = tf.transpose(z2_predict_input, perm=[1, 0])  # z2_predict = z2_predict_input.T  # 1xN
  L1_square = tf.transpose(L1_square_input, perm=[1, 0])  # L1_square = L1_square_input.T  # 1xN
  a = tf.square(f[0, :] * f[1, :]) + tf.square(p2[0, :] * f[1, :]) + tf.square(p2[1, :] * f[0, :])  # a = (f[0, :] * f[1, :])**2 + (p2[0, :] * f[1, :])**2 + (p2[1, :] * f[0, :])**2  # Nx1
  a = tf.reshape(a, [1, -1])  #  a = np.reshape(a, [1, -1])  # 1xN

  b = -2.0 * p1[0, :] * z1 * p2[0, :] * tf.square(f[1, :]) + 2.0 * tf.square(p2[0, :]) * z1 * tf.square(f[1, :]) - 2.0 * p1[1, :] \
      * z1 * p2[1, :] * tf.square(f[0, :]) + 2.0 * tf.square(p2[1, :]) * z1 * tf.square(f[0, :])

  c = tf.square(p1[0, :] * z1 * f[1, :]) - 2.0 * p1[0, :] * p2[0, :] * tf.square(z1 * f[1, :]) + tf.square(p2[0, :] * z1 * f[1, :]) \
      + tf.square(p1[1, :] * z1 * f[0, :]) - 2.0 * p1[1, :] * p2[1, :] * tf.square(z1 * f[0, :]) + tf.square(p2[1, :] * z1 * f[0, :]) \
      - L1_square * tf.square(f[0, :] * f[1, :])

  min_threshold = np.full(a.get_shape().as_list(), 1e-8)
  min_threshold = tf.constant(min_threshold, dtype=tf.float64)

  b_ac = tf.square(b) - 4.0 * a * c
  numer1 = tf.where(b_ac > tf.zeros_like(a, dtype=tf.float64), (-b + tf.sqrt(b_ac)),
  tf.zeros_like(a, dtype=tf.float64))
  numer2 = tf.where(b_ac > tf.zeros_like(a, dtype=tf.float64),
                    (-b - tf.sqrt(b_ac)),
                    tf.zeros_like(a, dtype=tf.float64))

  delta_z1 = tf.where(tf.abs(a) < min_threshold, tf.zeros_like(a, dtype=tf.float64), numer1 / (2.0 * a))
  delta_z2 = tf.where(tf.abs(a) < min_threshold, tf.zeros_like(a, dtype=tf.float64), numer2 / (2.0 * a))

  delta_z_predict = z2_predict - z1_predict

  delta_z = tf.where(delta_z1 * delta_z2 < 0, tf.where(z2_predict > z1_predict, delta_z1, delta_z2),
                     tf.where(tf.abs(delta_z1 - delta_z_predict) < tf.abs(delta_z2 - delta_z_predict), delta_z1,
                              delta_z2))

  z2 = z1 + delta_z  # 1xN
  x2_y2 = z2 * p2 / f  # 2xN
  recovered = tf.concat([x2_y2, z2], axis=0)  # recovered = np.concatenate((x2_y2, z2),axis=0)  # 3xN
  recovered = tf.transpose(recovered, perm=[1, 0])  # recovered = recovered.T

  delta_flag = tf.where(tf.abs(a) < min_threshold, tf.zeros_like(delta_z, dtype=tf.float64), tf.ones_like(delta_z, dtype=tf.float64))
  delta_flag2 = tf.where(b_ac > tf.zeros_like(a, dtype=tf.float64), tf.ones_like(a, dtype=tf.float64),
                        tf.zeros_like(a, dtype=tf.float64))
  delta_flag = delta_flag * delta_flag2
  delta_flag = tf.transpose(delta_flag, perm=[1, 0])

  return recovered, delta_flag

def back_project_skeleton_to_world(P_prediction, P_image, f, c, L1_square, depth_prediction):

  B = tf.shape(P_prediction)[0]
  valid_flag = tf.ones([B, 1], dtype=tf.float64)


  z1_original = tf.reshape(depth_prediction, [-1, 1])
  z1 = z1_original

  hip_image = P_image[:, 0, :]
  X = hip_image - c  # Nx2

  XX = z1 * X  # Nx2
  xy = XX / f
  xyz = tf.concat((xy, z1), axis=1)  # xyz = np.concatenate((xy, z1), axis=1)  # Nx3
  xyz = tf.expand_dims(xyz, 1)  # [Nx1x3]

  for i in range(3):
    p1 = tf.reshape(P_image[:, i, :], [-1, 2])  # p1 = np.reshape(P_image[:, i, :], [-1, 2])  # Nx2
    p2 = tf.reshape(P_image[:, i + 1, :], [-1, 2])  # p2 = np.reshape(P_image[:, i + 1, :], [-1, 2])  # Nx2
    z1_predict = tf.reshape(P_prediction[:, i, 2], [-1, 1])  # z1_predict = np.reshape(P_prediction[:, i, 2], [-1, 1])  # Nx1
    z2_predict = tf.reshape(P_prediction[:, i + 1, 2], [-1, 1])  # z2_predict = np.reshape(P_prediction[:, i+1, 2], [-1, 1])  # Nx1
    L1_segment_square = tf.reshape(L1_square[:, i], [-1, 1])  # L1_segment_square = np.reshape(L1_square[:, i], [-1, 1])  # Nx1
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  #Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)  # P_recovered[:, i + 1, :] = p_recovered
    valid_flag = valid_flag * delta_flag

  z1 = z1_original
  lower_joint = [0, 4, 5]
  upper_joint = [4, 5, 6]

  for i in range(3):
    p1 = tf.reshape(P_image[:, lower_joint[i], :], [-1, 2])
    p2 = tf.reshape(P_image[:, upper_joint[i], :], [-1, 2])
    z1_predict = tf.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])
    z2_predict = tf.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])
    L1_segment_square = tf.reshape(L1_square[:, 3 + i], [-1, 1])
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)
    valid_flag = valid_flag * delta_flag

  z1 = z1_original
  lower_joint = [0, 7]
  upper_joint = [7, 8]

  for i in range(2):
    p1 = tf.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = tf.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = tf.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = tf.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = tf.reshape(L1_square[:, 6 + i], [-1, 1])
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)
    valid_flag = valid_flag * delta_flag

  z1_original = z1

  z1 = z1_original
  lower_joint = [8, 9]
  upper_joint = [9, 10]

  for i in range(2):
    p1 = tf.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = tf.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = tf.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = tf.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = tf.reshape(L1_square[:, 8 + i], [-1, 1])
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)
    valid_flag = valid_flag * delta_flag

  z1 = z1_original
  lower_joint = [8, 11, 12]
  upper_joint = [11, 12, 13]

  for i in range(3):
    p1 = tf.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = tf.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = tf.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = tf.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = tf.reshape(L1_square[:, 10 + i], [-1, 1])
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)
    valid_flag = valid_flag * delta_flag

  z1 = z1_original
  lower_joint = [8, 14, 15]
  upper_joint = [14, 15, 16]

  for i in range(3):
    p1 = tf.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # p1 = np.reshape(P_image[:, lower_joint[i], :], [-1, 2])  # Nx2
    p2 = tf.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # p2 = np.reshape(P_image[:, upper_joint[i], :], [-1, 2])  # Nx2
    z1_predict = tf.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # z1_predict = np.reshape(P_prediction[:, lower_joint[i], 2], [-1, 1])  # Nx1
    z2_predict = tf.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # z2_predict = np.reshape(P_prediction[:, upper_joint[i], 2], [-1, 1])  # Nx1
    L1_segment_square = tf.reshape(L1_square[:, 13 + i], [-1, 1])
    p_recovered, delta_flag = back_project_point_to_world(p1, p2, f, c, z1, z1_predict, z2_predict, L1_segment_square)  # Nx3
    z1 = p_recovered[:, 2]
    z1 = tf.reshape(z1, [-1, 1])
    xyz = tf.concat((xyz, tf.expand_dims(p_recovered, 1)), axis=1)
    valid_flag = valid_flag * delta_flag


  return xyz, valid_flag