# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
# import cupy as np
# import minpy.numpy as np
import os
import cameras
import procrustes
import struct
from increase_std_avg import *
from LiftLearner import LiftLearner
from data_loader import DataLoader
import tensorflow.contrib.slim as slim
from multiprocessing.dummy import Pool as ThreadPool

flags = tf.app.flags

flags.DEFINE_float("z2_relative_mean", -0.9814330608182875, "the mean of z2 related to root joint")
flags.DEFINE_float("z3_relative_mean", 71.93292692607659, "the mean of z3 of samples")
flags.DEFINE_float("z4_relative_mean", 155.35298624679604, "the mean of z4 of samples")
flags.DEFINE_float("z5_relative_mean", 0.9814238622052575, "the mean of z5 of samples")
flags.DEFINE_float("z6_relative_mean", 72.22580949538178, "the mean of z6 of samples")
flags.DEFINE_float("z7_relative_mean", 158.01640792497318, "the mean of z7 of samples")
flags.DEFINE_float("z8_relative_mean", -47.421091513316604, "the mean of z8 of samples")
flags.DEFINE_float("z9_relative_mean", -97.69740161251556, "the mean of z9 of samples")
flags.DEFINE_float("z10_relative_mean", -110.66661784124814, "the mean of z10 of samples")
flags.DEFINE_float("z11_relative_mean", -131.17140037750661, "the mean of z11 of samples")
flags.DEFINE_float("z12_relative_mean", -86.81454165862303, "the mean of z12 of samples")
flags.DEFINE_float("z13_relative_mean", -43.453693420073996, "the mean of z13 of samples")
flags.DEFINE_float("z14_relative_mean", -28.252604877351022, "the mean of z14 of samples")
flags.DEFINE_float("z15_relative_mean", -85.51257835096797, "the mean of z15 of samples")
flags.DEFINE_float("z16_relative_mean", -43.136836370233695, "the mean of z16 of samples")
flags.DEFINE_float("z17_relative_mean", -37.39024982987547, "the mean of z1 of samples")


# for all smaple
flags.DEFINE_float("z1_mean", 5147.57798, "the mean of z1 of samples")
flags.DEFINE_float("z2_mean", 5146.596544714017, "the mean of z2 of samples")
flags.DEFINE_float("z3_mean", 5219.510904700912, "the mean of z3 of samples")
flags.DEFINE_float("z4_mean", 5302.930964021631, "the mean of z4 of samples")
flags.DEFINE_float("z5_mean", 5148.55940163704, "the mean of z5 of samples")
flags.DEFINE_float("z6_mean", 5219.803787270218, "the mean of z6 of samples")
flags.DEFINE_float("z7_mean", 5305.594385699808, "the mean of z7 of samples")
flags.DEFINE_float("z8_mean", 5100.156886261518, "the mean of z8 of samples")
flags.DEFINE_float("z9_mean", 5049.880576162319, "the mean of z9 of samples")
flags.DEFINE_float("z10_mean", 5036.911359933587, "the mean of z10 of samples")
flags.DEFINE_float("z11_mean", 5016.406577397328, "the mean of z11 of samples")
flags.DEFINE_float("z12_mean", 5060.763436116213, "the mean of z12 of samples")
flags.DEFINE_float("z13_mean", 5104.124284354761, "the mean of z13 of samples")
flags.DEFINE_float("z14_mean", 5119.325372897485, "the mean of z14 of samples")
flags.DEFINE_float("z15_mean", 5062.065399423867, "the mean of z15 of samples")
flags.DEFINE_float("z16_mean", 5104.441141404602, "the mean of z16 of samples")
flags.DEFINE_float("z17_mean", 5110.18772794496, "the mean of z17 of samples")

flags.DEFINE_float("z1_std", 751.1238008303231, "the std of z1 of samples")
flags.DEFINE_float("z2_std", 788.192905740467, "the std of z2 of samples")
flags.DEFINE_float("z3_std", 807.1752714391871, "the std of z3 of samples")
flags.DEFINE_float("z4_std", 809.2488754343542, "the std of z4 of samples")
flags.DEFINE_float("z5_std", 719.4635767158886, "the std of z5 of samples")
flags.DEFINE_float("z6_std", 748.356606931121, "the std of z6 of samples")
flags.DEFINE_float("z7_std", 757.335658389996, "the std of z7 of samples")
flags.DEFINE_float("z8_std", 759.1184065214757, "the std of z8 of samples")
flags.DEFINE_float("z9_std", 772.6060994496744, "the std of z9 of samples")
flags.DEFINE_float("z10_std", 777.9709650353619, "the std of z10 of samples")
flags.DEFINE_float("z11_std", 780.8715911019083, "the std of z11 of samples")
flags.DEFINE_float("z12_std", 740.1488441625569, "the std of z12 of samples")
flags.DEFINE_float("z13_std", 722.1468380842769, "the std of z13 of samples")
flags.DEFINE_float("z14_std", 743.1846316768235, "the std of z14 of samples")
flags.DEFINE_float("z15_std", 806.353878563109, "the std of z15 of samples")
flags.DEFINE_float("z16_std", 843.3530684748407, "the std of z16 of samples")
flags.DEFINE_float("z17_std", 846.021040653582, "the std of z17 of samples")

flags.DEFINE_integer("batch_size", 28, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 299, "Image height")
flags.DEFINE_integer("img_width", 299, "Image width")
flags.DEFINE_integer("resize_img_height", 299, "Image height")
flags.DEFINE_integer("resize_img_width", 299, "Image width")
flags.DEFINE_string("dataset_dir", "..\\Human3.6M\\filled_resized_images", "Dataset directory")
flags.DEFINE_string("configuration_dir", "..\\data_configuration", "Configurate file directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", "./all_samples_saved_checkpoints/model-8318395", "Specific checkpoint file to initialize from")
flags.DEFINE_string("output_dir", "..\\data_configuration", "Output directory")
flags.DEFINE_boolean("use_batch_normal_flag", True, "Indicate the net use batch normal")
tf.app.flags.DEFINE_boolean("procrustes", True, "Apply procrustes analysis at test time")
tf.app.flags.DEFINE_boolean("eliminate_invalidate_samples", True, "Eliminate the invalidate samples while loss caculating")
tf.app.flags.DEFINE_boolean("use_GT_flag", False, "Estimate delta z according if ground truth is applied")
tf.app.flags.DEFINE_boolean("relative_z_prdicted", False, "Incidcate if the outhput of sceond network is the relative z or the z coordinates of joints")
tf.app.flags.DEFINE_boolean("sample_normal_flag", True, "")
tf.app.flags.DEFINE_boolean("two_network", False, "")
tf.app.flags.DEFINE_boolean("save_predict", True, "")
tf.app.flags.DEFINE_boolean("use_z1_prdicted", True, "")
flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
FLAGS = flags.FLAGS

joint_tree_struct = [
        [[1, [2, [3]]], [4, [5, [6]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]],
        [[2, [3]], [0, [[4, [5, [6]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]],
        [[3], [1, [0, [[4, [5, [6]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]]],
        [2, [1, [0, [[4, [5, [6]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]]],
        [[5, [6]], [0, [[1, [2, [3]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]],
        [[6], [4, [0, [[1, [2, [3]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]]],
        [5, [4, [0, [[1, [2, [3]]], [7, [8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]]]]]]],
        [[8, [[9, [10]], [11, [12, [13]]], [14, [15, [16]]]]], [0, [[1, [2, [3]]], [4, [5, [6]]]]]],
        [[9, [10]], [11, [12, [13]]], [14, [15, [16]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]],
        [[10], [8, [[7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]], [11, [12, [13]]], [14, [15, [16]]]]]],
        [9, [8, [[7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]], [11, [12, [13]]], [14, [15, [16]]]]]],
        [[12, [13]], [8, [[9, [10]], [14, [15, [16]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]],
        [[13], [11, [8, [[9, [10]], [14, [15, [16]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]]],
        [12, [11, [8, [[9, [10]], [14, [15, [16]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]]],
        [[15, [16]], [8, [[9, [10]], [11, [12, [13]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]],
        [[16], [14, [8, [[9, [10]], [11, [12, [13]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]]],
        [15, [14, [8, [[9, [10]], [11, [12, [13]]], [7, [0, [[1, [2, [3]]], [4, [5, [6]]]]]]]]]]
    ]

seg_tree_struct = [
    [[0, [1, [2]]], [3, [4, [5]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]],
    [[1, [2]], [0, [[3, [4, [5]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]],
    [[2], [1, [0, [[3, [4, [5]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]]],
    [2, [1, [0, [[3, [4, [5]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]]],
    [[4, [5]], [3, [[0, [1, [2]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]],
    [[5], [4, [3, [[0, [1, [2]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]]],
    [5, [4, [3, [[0, [1, [2]]], [6, [7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]]]]]]],
    [[7, [[8, [9]], [10, [11, [12]]], [13, [14, [15]]]]], [6, [[0, [1, [2]]], [3, [4, [5]]]]]],
    [[8, [9]], [10, [11, [12]]], [13, [14, [15]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]],
    [[9], [8, [[7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]], [10, [11, [12]]], [13, [14, [15]]]]]],
    [9, [8, [[7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]], [10, [11, [12]]], [13, [14, [15]]]]]],
    [[11, [12]], [10, [[8, [9]], [13, [14, [15]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]],
    [[12], [11, [10, [[8, [9]], [13, [14, [15]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]]],
    [12, [11, [10, [[8, [9]], [13, [14, [15]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]]],
    [[14, [15]], [13, [[8, [9]], [10, [11, [12]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]],
    [[15], [14, [13, [[8, [9]], [10, [11, [12]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]]],
    [15, [14, [13, [[8, [9]], [10, [11, [12]]], [7, [6, [[0, [1, [2]]], [3, [4, [5]]]]]]]]]]
]

f_list = np.zeros([8, 2])
c_list = np.zeros([8, 2])
L1_square_list = np.zeros([8, 16])
z_root_joint_list = [0]*17
z_relative_prediction_list = [0]*17
P_image_list = np.zeros([8,2])

def get_best_pose(idx):
    return cameras.get_back_project_skeleton_to_world(z_root_joint_list[idx], z_relative_prediction_list[idx], P_image_list,
                                               np.array(f_list), np.array(c_list), np.array(L1_square_list),
                                               idx,
                                               joint_tree_struct[idx],
                                               seg_tree_struct[idx])

def evaluate_batches_with_all_z(predicted_z, P_image, f, c, L1_square, P):
    z_gt = P[:, :, 2]
    z_diff = np.fabs(predicted_z - z_gt)
    z_diff_mean_temp = np.mean(z_diff)
    z1_diff = np.fabs(predicted_z[:, 0] - z_gt[:, 0])

    batch_size = predicted_z.shape[0]
    valid_flag = np.zeros([batch_size], dtype=np.float64)
    valid_P_recovered = np.zeros_like(P)
    valid_z_diff_mean = np.full([batch_size], 1e10)
    joint_idx = [i for i in range(predicted_z.shape[1])]  # [0, 1, 2, ..., 16]
    test_joint_idx = [0]  # [0, 7, 8]
    global P_image_list
    global f_list
    global c_list
    global L1_square_list

    P_image_list = P_image
    f_list = f
    c_list = c
    L1_square_list = L1_square
    for i in test_joint_idx:
        root_idx = i
        z_root_joint = predicted_z[:, root_idx]
        z_root_joint_list[i] = np.expand_dims(z_root_joint, axis=1) # [b, 1]

        not_root_idx = list(set(joint_idx).difference(set([root_idx])))
        z_except_root = predicted_z[:, not_root_idx]  # [N, 16]
        z_relative_prediction_list[i] = z_except_root - z_root_joint_list[i]   # [N, 16]

    pool = ThreadPool(len(test_joint_idx))
    recovered_return = pool.map(get_best_pose, test_joint_idx)
    pool.close()
    pool.join()
    for i in range(len(test_joint_idx)):

        temp_P_recovered = recovered_return[i][0]
        temp_valid_flag = recovered_return[i][1]
        temp_z_diff_mean = recovered_return[i][2]

        expand_temp_z_diff_mean = np.expand_dims(temp_z_diff_mean, axis=1)
        expand_temp_z_diff_mean = np.repeat(expand_temp_z_diff_mean, predicted_z.shape[1], axis=1)
        expand_temp_z_diff_mean = np.expand_dims(expand_temp_z_diff_mean, axis=2)
        expand_temp_z_diff_mean = np.repeat(expand_temp_z_diff_mean, 3, axis=2)

        if i ==0:
            P_recovered = temp_P_recovered
            z_diff_mean = temp_z_diff_mean
        else:
            expand_z_diff_mean = np.expand_dims(z_diff_mean, axis=1)
            expand_z_diff_mean = np.repeat(expand_z_diff_mean, predicted_z.shape[1], axis=1)
            expand_z_diff_mean = np.expand_dims(expand_z_diff_mean, axis=2)
            expand_z_diff_mean = np.repeat(expand_z_diff_mean, 3, axis=2)

            P_recovered = np.where(expand_temp_z_diff_mean < expand_z_diff_mean, temp_P_recovered, P_recovered)
            z_diff_mean = np.where(temp_z_diff_mean < z_diff_mean, temp_z_diff_mean, z_diff_mean)
        

        expand_temp_valid_flag = np.expand_dims(temp_valid_flag, axis=1)
        expand_temp_valid_flag = np.repeat(expand_temp_valid_flag, predicted_z.shape[1], axis=1)
        expand_temp_valid_flag = np.expand_dims(expand_temp_valid_flag, axis=2)
        expand_temp_valid_flag = np.repeat(expand_temp_valid_flag, 3, axis=2)

        expand_valid_flag = np.expand_dims(valid_flag, axis=1)
        expand_valid_flag = np.repeat(expand_valid_flag, predicted_z.shape[1], axis=1)
        expand_valid_flag = np.expand_dims(expand_valid_flag, axis=2)
        expand_valid_flag = np.repeat(expand_valid_flag, 3, axis=2)

        expand_valid_z_diff_mean = np.expand_dims(valid_z_diff_mean, axis=1)
        expand_valid_z_diff_mean = np.repeat(expand_valid_z_diff_mean, predicted_z.shape[1], axis=1)
        expand_valid_z_diff_mean = np.expand_dims(expand_valid_z_diff_mean, axis=2)
        expand_valid_z_diff_mean = np.repeat(expand_valid_z_diff_mean, 3, axis=2)

        valid_P_recovered = np.where(expand_temp_valid_flag > 0.1, np.where(expand_valid_flag < 0.1, temp_P_recovered, np.where(expand_temp_z_diff_mean < expand_valid_z_diff_mean, temp_P_recovered, valid_P_recovered)), valid_P_recovered)
        valid_z_diff_mean = np.where(temp_valid_flag > 0.1, np.where(valid_flag < 0.1, temp_z_diff_mean, np.where(temp_z_diff_mean < valid_z_diff_mean, temp_z_diff_mean,
                                                                     valid_z_diff_mean)), valid_z_diff_mean)
        valid_flag = np.where(temp_valid_flag > 0.1, temp_valid_flag, valid_flag)

    valid_concerned_flag = True
    if valid_concerned_flag:

        expand_valid_flag_flag = np.expand_dims(valid_flag, axis=1)
        expand_valid_flag_flag = np.repeat(expand_valid_flag_flag, predicted_z.shape[1], axis=1)
        expand_valid_flag_flag = np.expand_dims(expand_valid_flag_flag, axis=2)
        expand_valid_flag_flag = np.repeat(expand_valid_flag_flag, 3, axis=2)
        P_recovered = np.where(expand_valid_flag_flag > 0.1, valid_P_recovered, P_recovered)  # indicate that at least one sample is valid
    valid_threshold = False
    if FLAGS.eliminate_invalidate_samples:
        if valid_threshold:
            valid_zero = np.zeros([batch_size], dtype=np.float64)
            valid_flag = np.where(np.sqrt(valid_z_diff_mean) > 20, valid_zero, valid_flag)
        P_recovered_relative_return = np.zeros([batch_size, 16, 3], dtype=np.float64)
        valid_flag = np.nonzero(valid_flag)
        P_recovered = P_recovered[valid_flag[0], :]
        P = P[valid_flag[0], :]
        if len(P) == 0:
            if FLAGS.procrustes:
                return [], [], [], P_recovered_relative_return, []
            else:
                return [], [], P_recovered_relative_return, []

        P_recovered_relative_return[valid_flag[0], :] = P_recovered[:, 1:, :] - np.reshape(P_recovered[:, 0, :],
                                                                                          [-1, 1, 3])
    else:
        P_recovered_relative_return = P_recovered[:, 1:, :] - np.reshape(P_recovered[:, 0, :],
                                                                         [-1, 1, 3])


    P_gt_relative = P[:, 1:, :] - np.reshape(P[:, 0, :], [-1, 1, 3])
    P_recovered_relative = P_recovered[:, 1:, :] - np.reshape(P_recovered[:, 0, :], [-1, 1, 3])
    P_recovered_relative_transform = np.zeros_like(P_recovered_relative)

    # Apply per-frame procrustes alignment if asked to do so
    if FLAGS.procrustes:
        for j in range(P_gt_relative.shape[0]):
            gt = P_gt_relative[j, :, :]  # [16, 3]
            out = P_recovered_relative[j, :, :]
            _, Z, T, b, c = procrustes.compute_similarity_transform(gt, out, compute_optimal_scale=True)
            out = (b * out.dot(T)) + c

            P_recovered_relative_transform[j, :, :] = out  # np.reshape(out, [-1, 3])

            sqerr_transform = (P_recovered_relative_transform - P_gt_relative) ** 2
            dist_transform = np.sqrt(np.sum(sqerr_transform, axis=2))
            total_err_transform = np.mean(dist_transform)
            dist_transform = np.reshape(dist_transform, [dist_transform.shape[0] * dist_transform.shape[1]]).tolist()

    # Compute Euclidean distance error per joint
    sqerr = (P_recovered_relative - P_gt_relative) ** 2
    dist = np.sqrt(np.sum(sqerr, axis=2))
    # print(np.mean(dist,axis=1))
    total_err = np.mean(dist)
    dist = np.reshape(dist, [dist.shape[0] * dist.shape[1]]).tolist()

    child_joints = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    father_joints = [i for i in range(1, 17)]
    P_recovered_relative_to_father = P_recovered[:, father_joints, :] - P_recovered[:, child_joints, :]
    P_gt_relative_to_father = P[:, father_joints, :] - P[:, child_joints, :]
    sqerr_relative = (P_recovered_relative_to_father - P_gt_relative_to_father) ** 2
    dist_relative = np.sqrt(np.sum(sqerr_relative, axis=2))
    total_err_relative = np.mean(dist_relative)
    dist_relative = np.reshape(dist_relative, [dist_relative.shape[0] * dist_relative.shape[1]]).tolist()
    if FLAGS.procrustes:
        print("error, transformed error,relative error and z1_diff of batch are {} {} {} and {}".format(total_err,
                                                                                            total_err_transform,
                                                                                            total_err_relative,
                                                                                                       z_diff_mean_temp))
        return dist, dist_transform, dist_relative, P_recovered_relative_return, z1_diff
    else:
        print("error,relative error and z1_diff of batch are {} {} and {}".format(total_err, total_err_relative, z_diff_mean_temp))
        return dist, dist_relative, P_recovered_relative_return, z1_diff

def evaluate_batches(z_root_joint, z_relative_prediction, P_image, f, c, L1_square, P):

    root_idx = 0
    P_recovered, valid_flag, _ = cameras.get_back_project_skeleton_to_world(z_root_joint, z_relative_prediction, P_image, np.array(f), np.array(c), np.array(L1_square), root_idx,
                                                             joint_tree_struct[root_idx], seg_tree_struct[root_idx])

    if FLAGS.eliminate_invalidate_samples:
        valid_flag = np.nonzero(valid_flag)
        P_recovered = P_recovered[valid_flag[0], :]
        P = P[valid_flag[0], :]
        if len(P) == 0:
            if FLAGS.procrustes:
                return [], [], []
            else:
                return [], []

    P_gt_relative = P[:, 1:, :] - np.reshape(P[:, 0, :], [-1, 1, 3])  # [N, 16, 3]
    P_recovered_relative = P_recovered[:, 1:, :] - np.reshape(P_recovered[:, 0, :], [-1, 1, 3])  # [N, 16, 3]
    P_recovered_relative_transform = np.zeros_like(P_recovered_relative)

    # Apply per-frame procrustes alignment if asked to do so
    if FLAGS.procrustes:
        for j in range(P_gt_relative.shape[0]):
            gt = P_gt_relative[j, :, :]  # [16, 3]
            out = P_recovered_relative[j, :, :]
            _, Z, T, b, c = procrustes.compute_similarity_transform(gt, out, compute_optimal_scale=True)
            out = (b * out.dot(T)) + c

            P_recovered_relative_transform[j, :, :] = out  # np.reshape(out, [-1, 3])

            sqerr_transform = (P_recovered_relative_transform - P_gt_relative) ** 2  # [N, 16, 3]Squared error between prediction and expected output
            dist_transform = np.sqrt(np.sum(sqerr_transform, axis=2))  # [N, 16]
            total_err_transform = np.mean(dist_transform)
            dist_transform = np.reshape(dist_transform, [dist_transform.shape[0] * dist_transform.shape[1]]).tolist()

    # Compute Euclidean distance error per joint
    sqerr = (P_recovered_relative - P_gt_relative) ** 2  # [N, 16, 3]Squared error between prediction and expected output
    dist = np.sqrt(np.sum(sqerr, axis=2))  # [N, 16]
    total_err = np.mean(dist)
    dist = np.reshape(dist, [dist.shape[0] * dist.shape[1]]).tolist()

    child_joints = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    father_joints = [i for i in range(1, 17)]
    P_recovered_relative_to_father = P_recovered[:, father_joints, :] - P_recovered[:, child_joints, :]  # [N, 16, 3]
    P_gt_relative_to_father = P[:, father_joints, :] - P[:, child_joints, :]  # [N, 16, 3]
    sqerr_relative = (P_recovered_relative_to_father - P_gt_relative_to_father) ** 2
    dist_relative = np.sqrt(np.sum(sqerr_relative, axis=2))  # [N, 17]
    total_err_relative = np.mean(dist_relative)
    dist_relative = np.reshape(dist_relative, [dist_relative.shape[0] * dist_relative.shape[1]]).tolist()
    if FLAGS.procrustes:
        print("error, transformed error and relative error of batch are {} {} and {}".format(total_err, total_err_transform, total_err_relative))
        return dist, dist_transform, dist_relative
    else:
        print("error and relative error of batch are {} and {}".format(total_err,total_err_relative))
        return dist, dist_relative
def save_predict(predicted_3d_file_batch, predicted_3d_pose):
    for b in range(FLAGS.batch_size):
        with open(predicted_3d_file_batch[b], 'wb')as fp:
            for i in range(16):
                for j in range(3):
                    a = struct.pack('d', predicted_3d_pose[b][i][j])
                    fp.write(a)
def main(_):
    z_relative_mean = [FLAGS.z2_relative_mean,FLAGS.z3_relative_mean,FLAGS.z4_relative_mean,FLAGS.z5_relative_mean,
                       FLAGS.z6_relative_mean,FLAGS.z7_relative_mean,FLAGS.z8_relative_mean,FLAGS.z9_relative_mean,
                       FLAGS.z10_relative_mean,FLAGS.z11_relative_mean,FLAGS.z12_relative_mean,FLAGS.z13_relative_mean,
                       FLAGS.z14_relative_mean,FLAGS.z15_relative_mean,FLAGS.z16_relative_mean,FLAGS.z17_relative_mean]
    z_relative_mean = np.array(z_relative_mean)
    z_relative_mean = np.expand_dims(z_relative_mean, axis=0)  # [1, 16]

    z_mean = [FLAGS.z1_mean,FLAGS.z2_mean,FLAGS.z3_mean,FLAGS.z4_mean,
                       FLAGS.z5_mean,FLAGS.z6_mean,FLAGS.z7_mean,FLAGS.z8_mean,
                       FLAGS.z9_mean,FLAGS.z10_mean,FLAGS.z11_mean,FLAGS.z12_mean,
                       FLAGS.z13_mean,FLAGS.z14_mean,FLAGS.z15_mean,FLAGS.z16_mean,FLAGS.z17_mean]
    z_mean = np.array(z_mean)
    z_mean = np.expand_dims(z_mean, axis=0)  # [1, 17]

    z_std = [FLAGS.z1_std, FLAGS.z2_std, FLAGS.z3_std, FLAGS.z4_std,
              FLAGS.z5_std, FLAGS.z6_std, FLAGS.z7_std, FLAGS.z8_std,
              FLAGS.z9_std, FLAGS.z10_std, FLAGS.z11_std, FLAGS.z12_std,
              FLAGS.z13_std, FLAGS.z14_std, FLAGS.z15_std, FLAGS.z16_std, FLAGS.z17_std]
    z_std = np.array(z_std)
    z_std = np.expand_dims(z_std, axis=0)  # [1, 17]
    z_std = np.repeat(z_std, FLAGS.batch_size, axis=0) # [b, 17]


    basename = os.path.basename(FLAGS.init_checkpoint_file)
    depth_output_file = FLAGS.output_dir + '/' + basename +'_depth'
    ordinal_output_file = FLAGS.output_dir + '/' + basename + '_ordinal'

    lift = LiftLearner(is_training=False)
    lift.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        linear_size=FLAGS.linear_size,
                        batch_size=FLAGS.batch_size,
                        use_batch_normal_flag = FLAGS.use_batch_normal_flag,
                        two_network = FLAGS.two_network)
    saver = tf.train.Saver([var for var in slim.get_model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    loader = DataLoader(FLAGS.dataset_dir,
                        FLAGS.configuration_dir,
                        FLAGS.batch_size,
                        FLAGS.img_height,
                        FLAGS.img_width,
                        FLAGS.resize_img_height,
                        FLAGS.resize_img_width,
                        FLAGS.use_GT_flag
                        )
    # for all
    val_dir = ['val - Directions.txt','val - Discussion.txt', 'val - Eating.txt', 'val - Greeting.txt', 'val - Phoning.txt', 'val - Photo.txt',\
               'val - Posing.txt', 'val - Purchases.txt', 'val - Sitting.txt', 'val - SittingDown.txt', 'val - Smoking.txt', 'val - Waiting.txt',\
               'val - WalkDog.txt', 'val - Walking.txt', 'val - WalkTogether.txt']
    act_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', \
               'Posing', 'Purchases', 'Sitting', 'SittingDown','Smoking', 'Waiting', \
               'WalkDog', 'Walking', 'WalkTogether']
    act_num =15

    loader.init_test_batch(val_dir, len(val_dir))

    with tf.Session(config=config) as sess:
        if FLAGS.init_checkpoint_file is None:
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        else:
            checkpoint = FLAGS.init_checkpoint_file
        saver.restore(sess,checkpoint)
        pred_depth = []
        pred_ordinal =[]
        error = []
        transformed_error = []
        relative_error = []
        z1_error = []
        for i in range(act_num):
            inc_dist = incre_std_avg()
            inc_dist_transform = incre_std_avg()
            inc_dist_relative = incre_std_avg()
            inc_z1_dist = incre_std_avg()
            for t in range(0, loader.sample_num[i], FLAGS.batch_size):
                if t % 100 == 0:
                    print('processing %s: %d/%d' % (basename, t, loader.sample_num[i]))
                inputs, P, P_image, f, c, L1_square, P_predicted, predicted_z1, predicted_3d_file_batch = loader.get_next_batch(i)
                if len(inputs) <= 0:
                    break
                predicted_z1_float32 = predicted_z1.astype(np.float32)
                predicted_z1_float32 = np.reshape(predicted_z1_float32, [FLAGS.batch_size, 1])  # [b, 1]
                predicted_z1_float32 = np.tile(predicted_z1_float32, (1, 17))  # [b, 17]

                inputs = inputs.astype(np.float32)
                f_float32 = f.astype(np.float32)
                c_float32 = c.astype(np.float32)
                P_image_float32 = P_image.astype(np.float32)
                P_image_float32 = np.reshape(P_image_float32, [FLAGS.batch_size, -1])
                L1_square_float32 = L1_square.astype(np.float32)
                L1_float32 = np.sqrt(L1_square_float32)
                L1_float32 = np.reshape(L1_float32, [FLAGS.batch_size, -1])

                joint_inputs = np.concatenate([f_float32, c_float32, P_image_float32, L1_float32],axis=1)
                pred = lift.inference(inputs, joint_inputs, sess)

                if FLAGS.relative_z_prdicted:
                    z_relative_pred = pred['ordinal'][:, 1:]  # [b, 16]
                    z_relative_pred = z_relative_pred + z_relative_mean
                else:
                    z_pred = pred['ordinal']
                    print("z_pred is {}".format(np.mean(z_pred)))

                    if FLAGS.sample_normal_flag:
                        z_pred = np.multiply(z_pred, z_std) + z_mean  # [b, 17]
                    else:
                        z_pred = z_pred + z_mean # [b, 17]
                if FLAGS.use_z1_prdicted:
                    z_relative_pred = z_pred - np.tile(np.reshape(z_pred[:, 0], [-1, 1]),(1,17))    # [b, 17]
                    z_pred = z_relative_pred + predicted_z1_float32 # [b, 17]

                dist, dist_transform, dist_relative, predicted_3d_pose, dist_z1 = evaluate_batches_with_all_z(z_pred, P_image, f, c, L1_square, P)
                if FLAGS.procrustes:
                    inc_dist_transform.incre_in_list(dist_transform)
                if FLAGS.save_predict:
                    save_predict(predicted_3d_file_batch, predicted_3d_pose)
                inc_dist.incre_in_list(dist)
                inc_dist_relative.incre_in_list(dist_relative)
                inc_z1_dist.incre_in_list(dist_z1)
                if t % 500 == 0:
                    if FLAGS.procrustes:
                        print("Temp error，transformed error and relative error are {} {} and {}".format(inc_dist.avg,
                                                                                                   inc_dist_transform.avg,
                                                                                                   inc_dist_relative.avg))
                    else:
                        print("Temp errorand relative error are {} and {}".format(inc_dist.avg, inc_dist_relative.avg))
            error.append(inc_dist.avg)
            if FLAGS.procrustes:
                transformed_error.append(inc_dist_transform.avg)
            relative_error.append(inc_dist_relative.avg)
            z1_error.append(inc_z1_dist.avg)
            if FLAGS.procrustes:
                print("{}: The total error，transformed error, relative error and z1 are {} {} {} and {}".format(act_name[i], inc_dist.avg,inc_dist_transform.avg,inc_dist_relative.avg, inc_z1_dist.avg))
            else:
                print("{}: The total error, relative error and z1 are {} {} and {}".format(act_name[i],inc_dist.avg, inc_dist_relative.avg, inc_z1_dist.avg))
        error_np = np.hstack(error)
        error_avg = np.mean(error_np)
        transformed_error_np = np.hstack(transformed_error)
        transformed_error_avg = np.mean(transformed_error_np)
        relative_error_np = np.hstack(relative_error)
        relative_error_avg = np.mean(relative_error_np)
        z1_error_np = np.hstack(z1_error)
        z1_error_avg = np.mean(z1_error_np)
        distance_item = []
        for i in range(act_num):
            distance_item.append(act_name[i])
            distance_item.append(": ")
            distance_item.append(repr(error[i]))
            distance_item.append(",")
            if FLAGS.procrustes:
                distance_item.append(repr(transformed_error[i]))
                distance_item.append(",")
            distance_item.append(repr(relative_error[i]))
            distance_item.append("\n")
        distance_item.append("average: ")
        distance_item.append(repr(error_avg))
        if FLAGS.procrustes:
            # distance_item.append(act_name[i])
            distance_item.append(": ")
            distance_item.append(",")
            distance_item.append(repr(transformed_error_avg))
        distance_item.append(",")
        distance_item.append(repr(relative_error_avg))

        distance_item.append("\n")
        distance_item.append("z1:")
        distance_item.append("\n")
        for i in range(act_num):
            distance_item.append(act_name[i])
            distance_item.append(": ")
            distance_item.append(repr(z1_error[i]))
            distance_item.append("\n")
        distance_item.append("average: ")
        distance_item.append(repr(z1_error_avg))

        distance_file = FLAGS.output_dir + '/distance.txt'
        with open(distance_file, "w") as f:
            f.writelines(distance_item)

        np.save(depth_output_file, pred_depth)
        np.save(ordinal_output_file, pred_ordinal)
        print('finished!')

if __name__ == '__main__':
    tf.app.run()