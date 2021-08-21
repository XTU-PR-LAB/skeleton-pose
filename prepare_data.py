# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import struct
from increase_std_avg import *
import cameras
import test
import cv2
# from data_loader import *

def test_project(P, f, c, L1_square):

    f_T = np.expand_dims(np.array(f), axis=2)  # shape = [N, 2, 1]
    c_T = np.expand_dims(np.array(c), axis=2)  # shape = [N, 2, 1]
    P_image = cameras.project_skeleton_to_image(np.array(P), f_T, c_T)  # Nx17x2
    P_recovered = cameras.back_project_skeleton_to_world(np.array(P), P_image, np.array(f), np.array(c), np.array(L1_square))

    zeros_thrould = np.full(P_recovered.shape, 1.0e-6)
    correct_num = np.where(np.fabs(np.array(P) - P_recovered) > zeros_thrould, np.ones_like(P_recovered), np.zeros_like(P_recovered))
    correct_num = np.sum(correct_num)
    print("{} points error".format(correct_num))


def save_dir_file(file_name, save_dir):
    with open(file_name, "w") as dir_f:
        for key in save_dir.keys():
            txt_line = []
            txt_line.append(key)
            txt_line.append(" ")
            txt_line.append(str(save_dir[key]))
            txt_line.append("\n")
            dir_f.writelines(txt_line)

def create_test_dir(data_dir):
    subj = [1, 5, 6, 7, 8, 9, 11]
    for s in subj:
        for act in range(2, 17):
            for sub_act in range(1, 3):
                for camera in range(1, 5):
                    dir_name = os.path.join(data_dir,"s_{:0>2d}_act_{:0>2d}_subact_{:0>2d}_ca_{:0>2d}".format(s,act,sub_act,camera))
                    isExists = os.path.exists(dir_name)
                    if not isExists:
                        os.makedirs(dir_name)

# create_test_dir("data_configuration")
annotations_dir = "..\\Human3.6M\\annotations"
prepare_data_dir = "..\\data_configuration"
# create_test_dir("..\\subject_resized_images")
relate_root_flag = True
def prepare_data(data_dir, prepare_dir):

    subj = [1, 5, 6, 7, 8, 9, 11]
    train_subj = [1, 5, 6, 7, 8]

    test_project_flag = False
    skeleton_per_frm = True
    inc_std = incre_std_avg()

    inc_std_2 = incre_std_avg()
    inc_std_3 = incre_std_avg()
    inc_std_4 = incre_std_avg()
    inc_std_5 = incre_std_avg()
    inc_std_6 = incre_std_avg()
    inc_std_7 = incre_std_avg()
    inc_std_8 = incre_std_avg()
    inc_std_9 = incre_std_avg()
    inc_std_10 = incre_std_avg()
    inc_std_11 = incre_std_avg()
    inc_std_12 = incre_std_avg()
    inc_std_13 = incre_std_avg()
    inc_std_14 = incre_std_avg()
    inc_std_15 = incre_std_avg()
    inc_std_16 = incre_std_avg()
    inc_std_17 = incre_std_avg()

    inc_std_relative_2 = incre_std_avg()
    inc_std_relative_3 = incre_std_avg()
    inc_std_relative_4 = incre_std_avg()
    inc_std_relative_5 = incre_std_avg()
    inc_std_relative_6 = incre_std_avg()
    inc_std_relative_7 = incre_std_avg()
    inc_std_relative_8 = incre_std_avg()
    inc_std_relative_9 = incre_std_avg()
    inc_std_relative_10 = incre_std_avg()
    inc_std_relative_11 = incre_std_avg()
    inc_std_relative_12 = incre_std_avg()
    inc_std_relative_13 = incre_std_avg()
    inc_std_relative_14 = incre_std_avg()
    inc_std_relative_15 = incre_std_avg()
    inc_std_relative_16 = incre_std_avg()
    inc_std_relative_17 = incre_std_avg()


    train_dir = {}
    val_dir = {}
    for s in subj:
        inc_z_coord = []

        inc_z2_coord = []
        inc_z3_coord = []
        inc_z4_coord = []
        inc_z5_coord = []
        inc_z6_coord = []
        inc_z7_coord = []
        inc_z8_coord = []
        inc_z9_coord = []
        inc_z10_coord = []
        inc_z11_coord = []
        inc_z12_coord = []
        inc_z13_coord = []
        inc_z14_coord = []
        inc_z15_coord = []
        inc_z16_coord = []
        inc_z17_coord = []

        inc_relative_z2_coord = []
        inc_relative_z3_coord = []
        inc_relative_z4_coord = []
        inc_relative_z5_coord = []
        inc_relative_z6_coord = []
        inc_relative_z7_coord = []
        inc_relative_z8_coord = []
        inc_relative_z9_coord = []
        inc_relative_z10_coord = []
        inc_relative_z11_coord = []
        inc_relative_z12_coord = []
        inc_relative_z13_coord = []
        inc_relative_z14_coord = []
        inc_relative_z15_coord = []
        inc_relative_z16_coord = []
        inc_relative_z17_coord = []

        camera_pose = {}
        focal_len = {}
        center_point = {}
        with open(os.path.join(data_dir,'Human36M_subject{}_camera.json'.format(s)), 'r') as f:
            data = json.load(f)

            for camera_no in range(1,5):
                camera_item = []
                camera_value = data[str(camera_no)]
                R = camera_value['R']
                for i in range(3):
                    for j in range(3):
                        camera_item.append(R[i][j])

                t = camera_value['t']
                camera_pose[camera_no] = [R, t]
                for i in range(3):
                    camera_item.append(t[i])

                f = camera_value['f']
                focal_len[camera_no] = f
                for i in range(2):
                    camera_item.append(f[i])

                c = camera_value['c']
                center_point[camera_no] = c
                camera_item.append(c[0])
                camera_item.append(c[1])


                for act in range(2, 17):
                    for sub_act in range(1, 3):

                        with open(os.path.join(data_dir, 'Human36M_subject{}_joint_3d.json'.format(s)), 'r') as f:
                            joint_data = json.load(f)
                            act_joint_data = joint_data[str(act)]
                            sub_act_joint_data = act_joint_data[str(sub_act)]

                        for image_no in range(len(sub_act_joint_data)):
                            dir_name = os.path.join(prepare_dir,
                                                    "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}".format(s, act,
                                                                                                             sub_act,
                                                                                                             camera_no),"s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}_{4:0>6d}_cam.bin".format(s, act,
                                                                                                             sub_act,
                                                                                                             camera_no, 1 + image_no))
                            with open(dir_name, "wb") as camf:
                                for i in range(len(camera_item)):
                                    a = struct.pack('d', camera_item[i])
                                    camf.write(a)


        with open(os.path.join(data_dir, 'Human36M_subject{}_joint_3d.json'.format(s)), 'r') as f:
            data = json.load(f)
            for act_no in range(2, 17):
                act_data = data[str(act_no)]
                for sub_act_no in range(1, 3):
                    sub_act_data = act_data[str(sub_act_no)]

                    for train_camera in range(1, 5):
                        dir_name = os.path.join(
                            "s_{:0>2d}_act_{:0>2d}_subact_{:0>2d}_ca_{:0>2d}".format(s, act_no, sub_act_no,
                                                                                     train_camera))
                        if s in train_subj:
                            train_dir[dir_name] = len(sub_act_data)
                        else:
                            val_dir[dir_name] = len(sub_act_data)

                    segment_data = []
                    for no in range(len(sub_act_data)):
                        segment_data.append(sub_act_data[str(no)])
                    segment_data = np.array(segment_data)

                    upper_joint = [x for x in range(1, 17)]
                    lower_joint = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

                    if test_project_flag:
                        test_segment = []
                    segment_item = []
                    skeleton = []  # 16xN
                    for i in range(16):
                        segment = segment_data[:, upper_joint[i], :] - segment_data[:, lower_joint[i], :]
                        segment = segment ** 2
                        segment = segment.sum(axis=1)

                        if skeleton_per_frm:
                            skeleton.append(segment.tolist())
                        else:
                            segment_mean = np.mean(segment)
                            test_segment. append(segment_mean)
                            segment_item.append(segment_mean)

                    if skeleton_per_frm:
                        for i in range(len(skeleton[0])):
                            skeleton_item = []
                            for j in range(len(skeleton)):
                                skeleton_item.append(skeleton[j][i])  # skeleton_item.append(repr(skeleton[j][i]))

                            segment_item.append(skeleton_item)


                    for camera_no in range(1, 5):
                        for frame_no in range(len(sub_act_data)):
                            segment_file_name = os.path.join(prepare_dir,
                                                            "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}".format(
                                                                s, act_no,
                                                                sub_act_no,
                                                                camera_no),
                                                            "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}_{4:0>6d}_seg.bin".format(
                                                                s, act_no,
                                                                sub_act_no,
                                                                camera_no, frame_no + 1))
                            with open(segment_file_name, "wb") as segf:
                                if skeleton_per_frm:
                                    for j in range(16):
                                        a = struct.pack('d', segment_item[frame_no][j])
                                        segf.write(a)

                                else:
                                    for j in range(16):
                                        a = struct.pack('d', segment_item[j])
                                        segf.write(a)

                    if test_project_flag:
                        test_P = []
                        test_f = []
                        test_c = []
                        test_L1_square = []

                    for camera_no in range(1, 5):
                        R, T = camera_pose[camera_no]
                        R = np.array(R)
                        T = np.reshape(np.array(T),[3, 1])
                        for image_no in range(len(sub_act_data)):
                            camera_coord = cameras.world_to_camera_frame( np.reshape(np.array(sub_act_data[str(image_no)]), [-1, 3]), R, T)

                            pose3d_file_name = os.path.join(prepare_dir,
                                                    "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}".format(s, act_no,
                                                                                                                 sub_act_no,
                                                                                                                 camera_no),
                                                    "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}_{4:0>6d}_pose3d_cam.bin".format(
                                                        s, act_no,
                                                        sub_act_no,
                                                        camera_no, image_no + 1))

                            inc_z_coord.append(camera_coord[0][2])

                            inc_z2_coord.append(camera_coord[1][2])
                            inc_z3_coord.append(camera_coord[2][2])
                            inc_z4_coord.append(camera_coord[3][2])
                            inc_z5_coord.append(camera_coord[4][2])
                            inc_z6_coord.append(camera_coord[5][2])
                            inc_z7_coord.append(camera_coord[6][2])
                            inc_z8_coord.append(camera_coord[7][2])
                            inc_z9_coord.append(camera_coord[8][2])
                            inc_z10_coord.append(camera_coord[9][2])
                            inc_z11_coord.append(camera_coord[10][2])
                            inc_z12_coord.append(camera_coord[11][2])
                            inc_z13_coord.append(camera_coord[12][2])
                            inc_z14_coord.append(camera_coord[13][2])
                            inc_z15_coord.append(camera_coord[14][2])
                            inc_z16_coord.append(camera_coord[15][2])
                            inc_z17_coord.append(camera_coord[16][2])

                            if relate_root_flag:
                                inc_relative_z2_coord.append(camera_coord[1][2] - camera_coord[0][2])
                                inc_relative_z3_coord.append(camera_coord[2][2] - camera_coord[0][2])
                                inc_relative_z4_coord.append(camera_coord[3][2] - camera_coord[0][2])
                                inc_relative_z5_coord.append(camera_coord[4][2] - camera_coord[0][2])
                                inc_relative_z6_coord.append(camera_coord[5][2] - camera_coord[0][2])
                                inc_relative_z7_coord.append(camera_coord[6][2] - camera_coord[0][2])
                                inc_relative_z8_coord.append(camera_coord[7][2] - camera_coord[0][2])
                                inc_relative_z9_coord.append(camera_coord[8][2] - camera_coord[0][2])
                                inc_relative_z10_coord.append(camera_coord[9][2] - camera_coord[0][2])
                                inc_relative_z11_coord.append(camera_coord[10][2] - camera_coord[0][2])
                                inc_relative_z12_coord.append(camera_coord[11][2] - camera_coord[0][2])
                                inc_relative_z13_coord.append(camera_coord[12][2] - camera_coord[0][2])
                                inc_relative_z14_coord.append(camera_coord[13][2] - camera_coord[0][2])
                                inc_relative_z15_coord.append(camera_coord[14][2] - camera_coord[0][2])
                                inc_relative_z16_coord.append(camera_coord[15][2] - camera_coord[0][2])
                                inc_relative_z17_coord.append(camera_coord[16][2] - camera_coord[0][2])

                            else:
                                inc_relative_z2_coord.append(camera_coord[1][2]-camera_coord[0][2])
                                inc_relative_z3_coord.append(camera_coord[2][2]-camera_coord[1][2])
                                inc_relative_z4_coord.append(camera_coord[3][2]-camera_coord[2][2])
                                inc_relative_z5_coord.append(camera_coord[4][2]-camera_coord[0][2])
                                inc_relative_z6_coord.append(camera_coord[5][2]-camera_coord[4][2])
                                inc_relative_z7_coord.append(camera_coord[6][2]-camera_coord[5][2])
                                inc_relative_z8_coord.append(camera_coord[7][2]-camera_coord[0][2])
                                inc_relative_z9_coord.append(camera_coord[8][2]-camera_coord[7][2])
                                inc_relative_z10_coord.append(camera_coord[9][2]-camera_coord[8][2])
                                inc_relative_z11_coord.append(camera_coord[10][2]-camera_coord[9][2])
                                inc_relative_z12_coord.append(camera_coord[11][2]-camera_coord[8][2])
                                inc_relative_z13_coord.append(camera_coord[12][2]-camera_coord[11][2])
                                inc_relative_z14_coord.append(camera_coord[13][2]-camera_coord[12][2])
                                inc_relative_z15_coord.append(camera_coord[14][2]-camera_coord[8][2])
                                inc_relative_z16_coord.append(camera_coord[15][2]-camera_coord[14][2])
                                inc_relative_z17_coord.append(camera_coord[16][2]-camera_coord[15][2])


                            with open(pose3d_file_name, 'wb')as fp:
                                assert camera_coord.shape[0] == 17 and camera_coord.shape[1] == 3
                                for i in range(17):
                                    for j in range(3):
                                        a = struct.pack('d', camera_coord[i][j])
                                        fp.write(a)

                            image_coord = cameras.project_point_to_image(camera_coord, np.reshape(np.array(focal_len[camera_no]), [-1, 1]), np.reshape(np.array(center_point[camera_no]), [-1, 1]))  # 17x2
                            image2d_file_name = os.path.join(prepare_dir,
                                                            "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}".format(
                                                                s, act_no,
                                                                sub_act_no,
                                                                camera_no),
                                                            "s_{0:0>2d}_act_{1:0>2d}_subact_{2:0>2d}_ca_{3:0>2d}_{4:0>6d}_image_coord.bin".format(
                                                                s, act_no,
                                                                sub_act_no,
                                                                camera_no, image_no + 1))
                            with open(image2d_file_name, 'wb')as fp:
                                assert image_coord.shape[0] == 17 and image_coord.shape[1] == 2
                                for i in range(17):
                                    for j in range(2):
                                        a = struct.pack('d', image_coord[i][j])
                                        fp.write(a)

                            if test_project_flag and image_no == 30:
                                test_P.append(camera_coord.tolist())
                                test_f.append(focal_len[camera_no])
                                test_c.append(center_point[camera_no])
                                if skeleton_per_frm:
                                    test_L1_square.append(np.array(skeleton)[:, image_no])
                                else:
                                    test_L1_square.append(test_segment)
                    if test_project_flag:
                        test_project(test_P, test_f, test_c, test_L1_square)

        inc_std.incre_in_list(inc_z_coord)

        inc_std_2.incre_in_list(inc_z2_coord)
        inc_std_3.incre_in_list(inc_z3_coord)
        inc_std_4.incre_in_list(inc_z4_coord)
        inc_std_5.incre_in_list(inc_z5_coord)
        inc_std_6.incre_in_list(inc_z6_coord)
        inc_std_7.incre_in_list(inc_z7_coord)
        inc_std_8.incre_in_list(inc_z8_coord)
        inc_std_9.incre_in_list(inc_z9_coord)
        inc_std_10.incre_in_list(inc_z10_coord)
        inc_std_11.incre_in_list(inc_z11_coord)
        inc_std_12.incre_in_list(inc_z12_coord)
        inc_std_13.incre_in_list(inc_z13_coord)
        inc_std_14.incre_in_list(inc_z14_coord)
        inc_std_15.incre_in_list(inc_z15_coord)
        inc_std_16.incre_in_list(inc_z16_coord)
        inc_std_17.incre_in_list(inc_z17_coord)

        inc_std_relative_2.incre_in_list(inc_relative_z2_coord)
        inc_std_relative_3.incre_in_list(inc_relative_z3_coord)
        inc_std_relative_4.incre_in_list(inc_relative_z4_coord)
        inc_std_relative_5.incre_in_list(inc_relative_z5_coord)
        inc_std_relative_6.incre_in_list(inc_relative_z6_coord)
        inc_std_relative_7.incre_in_list(inc_relative_z7_coord)
        inc_std_relative_8.incre_in_list(inc_relative_z8_coord)
        inc_std_relative_9.incre_in_list(inc_relative_z9_coord)
        inc_std_relative_10.incre_in_list(inc_relative_z10_coord)
        inc_std_relative_11.incre_in_list(inc_relative_z11_coord)
        inc_std_relative_12.incre_in_list(inc_relative_z12_coord)
        inc_std_relative_13.incre_in_list(inc_relative_z13_coord)
        inc_std_relative_14.incre_in_list(inc_relative_z14_coord)
        inc_std_relative_15.incre_in_list(inc_relative_z15_coord)
        inc_std_relative_16.incre_in_list(inc_relative_z16_coord)
        inc_std_relative_17.incre_in_list(inc_relative_z17_coord)

        if s == 8:
            print(inc_std.avg)
            print(inc_std.std)
            print("\n")
            print(inc_std_2.avg)
            print(inc_std_2.std)
            print("\n")
            print(inc_std_3.avg)
            print(inc_std_3.std)
            print("\n")
            print(inc_std_4.avg)
            print(inc_std_4.std)
            print("\n")
            print(inc_std_5.avg)
            print(inc_std_5.std)
            print("\n")
            print(inc_std_6.avg)
            print(inc_std_6.std)
            print("\n")
            print(inc_std_7.avg)
            print(inc_std_7.std)
            print("\n")
            print(inc_std_8.avg)
            print(inc_std_8.std)
            print("\n")
            print(inc_std_9.avg)
            print(inc_std_9.std)
            print("\n")
            print(inc_std_10.avg)
            print(inc_std_10.std)
            print("\n")
            print(inc_std_11.avg)
            print(inc_std_11.std)
            print("\n")
            print(inc_std_12.avg)
            print(inc_std_12.std)
            print("\n")
            print(inc_std_13.avg)
            print(inc_std_13.std)
            print("\n")
            print(inc_std_14.avg)
            print(inc_std_14.std)
            print("\n")
            print(inc_std_15.avg)
            print(inc_std_15.std)
            print("\n")
            print(inc_std_16.avg)
            print(inc_std_16.std)
            print("\n")
            print(inc_std_17.avg)
            print(inc_std_17.std)
            print("\n\n")
            print("\n")
            print(inc_std_relative_2.avg)
            print(inc_std_relative_2.std)
            print("\n")
            print(inc_std_relative_3.avg)
            print(inc_std_relative_3.std)
            print("\n")
            print(inc_std_relative_4.avg)
            print(inc_std_relative_4.std)
            print("\n")
            print(inc_std_relative_5.avg)
            print(inc_std_relative_5.std)
            print("\n")
            print(inc_std_relative_6.avg)
            print(inc_std_relative_6.std)
            print("\n")
            print(inc_std_relative_7.avg)
            print(inc_std_relative_7.std)
            print("\n")
            print(inc_std_relative_8.avg)
            print(inc_std_relative_8.std)
            print("\n")
            print(inc_std_relative_9.avg)
            print(inc_std_relative_9.std)
            print("\n")
            print(inc_std_relative_10.avg)
            print(inc_std_relative_10.std)
            print("\n")
            print(inc_std_relative_11.avg)
            print(inc_std_relative_11.std)
            print("\n")
            print(inc_std_relative_12.avg)
            print(inc_std_relative_12.std)
            print("\n")
            print(inc_std_relative_13.avg)
            print(inc_std_relative_13.std)
            print("\n")
            print(inc_std_relative_14.avg)
            print(inc_std_relative_14.std)
            print("\n")
            print(inc_std_relative_15.avg)
            print(inc_std_relative_15.std)
            print("\n")
            print(inc_std_relative_16.avg)
            print(inc_std_relative_16.std)
            print("\n")
            print(inc_std_relative_17.avg)
            print(inc_std_relative_17.std)


            z_mean_item = []
            z_mean_item.append(repr(inc_std.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_2.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_2.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_3.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_3.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_4.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_4.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_5.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_5.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_6.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_6.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_7.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_7.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_8.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_8.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_9.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_9.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_10.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_10.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_11.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_11.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_12.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_12.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_13.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_13.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_14.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_14.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_15.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_15.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_16.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_16.std))
            z_mean_item.append("\n")

            z_mean_item.append(repr(inc_std_17.avg))
            z_mean_item.append(",")
            z_mean_item.append(repr(inc_std_17.std))
            z_mean_item.append("\n")

            with open(os.path.join(prepare_dir,"z_mean_std.txt"), "w") as f:
                f.writelines(z_mean_item)

            z_relative_mean_item = []
            z_relative_mean_item.append(repr(inc_std_relative_2.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_2.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_3.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_3.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_4.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_4.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_5.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_5.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_6.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_6.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_7.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_7.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_8.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_8.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_9.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_9.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_10.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_10.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_11.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_11.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_12.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_12.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_13.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_13.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_14.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_14.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_15.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_15.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_16.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_16.std))
            z_relative_mean_item.append("\n")

            z_relative_mean_item.append(repr(inc_std_relative_17.avg))
            z_relative_mean_item.append(",")
            z_relative_mean_item.append(repr(inc_std_relative_17.std))
            z_relative_mean_item.append("\n")

            if relate_root_flag:
                with open(os.path.join(prepare_dir, "z_relative_root_mean_std.txt"), "w") as f:
                    f.writelines(z_relative_mean_item)
            else:
                with open(os.path.join(prepare_dir, "z_relative_father_mean_std.txt"), "w") as f:
                    f.writelines(z_relative_mean_item)

    save_dir_file(os.path.join(prepare_data_dir, "train.txt"), train_dir)
    save_dir_file(os.path.join(prepare_data_dir, "val.txt"), val_dir)
    print("finish!")

def format_image_file_list(data_root, resized_data_root, configuration_root):
    with open(os.path.join(configuration_root, "train.txt")) as f:
        frames = f.readlines()
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]
    frame_ids = [int(i) for i in frame_ids]
    image_file_list = []
    resized_image_file_list = []
    for i in range(len(frame_ids)):

        image_file_list.append([os.path.join(data_root, subfolders[i],
                                               subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                  range(1, 1 + frame_ids[i])])

        resized_image_file_list.append([os.path.join(resized_data_root, subfolders[i],
                                             subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                range(1, 1 + frame_ids[i])])

    image_file_list = [i for item in image_file_list for i in item]
    resized_image_file_list = [i for item in resized_image_file_list for i in item]

    return image_file_list, resized_image_file_list

dataset_dir = '..\\Human3.6M\\images'
resized_data_root  = '..\\Human3.6M\\resized_images'
configuration_dir = '..\\data_configuration'
image_height = 299
image_width = 299
def preprocess(images_name, resized_image_name):
    images_resized = [] #final result
    inc_image = []
    inc_std = incre_std_avg()
    # Resize and compute mean!
    for i in range(len(images_name)):
        X = cv2.imread(images_name[i])
        X = cv2.resize(X, (image_height, image_width))
        # images_resized.append(X)
        cv2.imwrite(resized_image_name[i], X)
        inc_std.incre_in_value(X)

    mean = inc_std.avg

    with open(os.path.join(configuration_dir, 'mean_image.bin'), 'wb')as fp:
        for i in range(image_height):
            for j in range(image_width):
                for k in range(3):
                    a = struct.pack('f', mean[i,j,k])
                    fp.write(a)
    read_mean_image(os.path.join(configuration_dir, 'mean_image.bin'))
    print('finished!')


def read_mean_image(file_name):
    image_mean = np.zeros((image_height, image_width, 3),dtype=np.float32)
    with open(file_name, "rb") as binf:
        for i in range(image_height):
            for j in range(image_width):
                # for k in range(3):
                temp = binf.read(4 * 3)
                data_raw = struct.unpack('f'* 3, temp)
                data_raw = np.reshape(np.array(data_raw), [1, -1])
                image_mean[i,j,:] = data_raw


def preprecoess_image():
    file_list, resized_file_list = format_image_file_list(dataset_dir, resized_data_root, configuration_dir)
    preprocess(file_list, resized_file_list)


def format_file_list(data_root, resized_data_root, resized_subject_data_root, configuration_root,file_name):
    with open(os.path.join(configuration_root, file_name)) as f:
        frames = f.readlines()
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]
    frame_ids = [int(i) for i in frame_ids]
    cam_file_list = []
    coord_3d_file_list = []
    coord_2d_file_list = []
    segment_file_list = []
    image_file_list = []
    resized_image_file_list = []
    resized_subject_image_file_list = []

    for i in range(len(frame_ids)):
        cam_file_list.append([os.path.join(configuration_root, subfolders[i],
                                           subfolders[i] + "_" + "{0:0>6d}".format(j) + "_cam.bin") for j in
                              range(1, 1 + frame_ids[i])])

        coord_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                subfolders[i] + "_" + "{0:0>6d}".format(j) + "_pose3d_cam.bin")
                                   for j in range(1, 1 + frame_ids[i])])

        coord_2d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                subfolders[i] + "_" + "{0:0>6d}".format(j) + "_image_coord.bin")
                                   for j in
                                   range(1, 1 + frame_ids[i])])
        segment_file_list.append([os.path.join(configuration_root, subfolders[i],
                                               subfolders[i] + "_" + "{0:0>6d}".format(j) + "_seg.bin") for j in
                                  range(1, 1 + frame_ids[i])])
        image_file_list.append([os.path.join(data_root, subfolders[i],
                                             subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                range(1, 1 + frame_ids[i])])
        resized_image_file_list.append([os.path.join(resized_data_root, subfolders[i],
                                                     subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                        range(1, 1 + frame_ids[i])])
        resized_subject_image_file_list.append([os.path.join(resized_subject_data_root, subfolders[i],
                                                     subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                        range(1, 1 + frame_ids[i])])

    cam_file_list = [i for item in cam_file_list for i in item]
    coord_3d_file_list = [i for item in coord_3d_file_list for i in item]

    coord_2d_file_list = [i for item in coord_2d_file_list for i in item]
    segment_file_list = [i for item in segment_file_list for i in item]
    image_file_list = [i for item in image_file_list for i in item]
    resized_image_file_list = [i for item in resized_image_file_list for i in item]
    resized_subject_image_file_list = [i for item in resized_subject_image_file_list for i in item]

    all_list = {}
    all_list['cam_file_list'] = cam_file_list
    all_list['coord_3d_file_list'] = coord_3d_file_list
    all_list['coord_2d_file_list'] = coord_2d_file_list
    all_list['segment_file_list'] = segment_file_list
    all_list['image_file_list'] = image_file_list
    all_list['resized_image_file_list'] = resized_image_file_list
    all_list['resized_subject_image_file_list'] = resized_subject_image_file_list

    return all_list

def gather(self, original, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
                         index.shape[dim + 1:]
    self_xsection_shape = original.shape[:dim] + original.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")

    data_swaped = np.swapaxes(original, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def get_bound_box(coord_2d):
    x_coord = coord_2d[:, 0]
    y_coord = coord_2d[:, 1]
    arg_min_x = np.argmin(x_coord)
    arg_max_x = np.argmax(x_coord)
    arg_min_y = np.argmin(y_coord)
    arg_max_y = np.argmax(y_coord)
    min_x = int(x_coord[arg_min_x])
    max_x = int(x_coord[arg_max_x])
    min_y = int(y_coord[arg_min_y])
    max_y = int(y_coord[arg_max_y])
    return [min_y, max_y, min_x, max_x]  # [arg_min_x, arg_min_y, arg_max_x - arg_min_x, arg_max_y - arg_min_y]

def get_filled_img(img, bb):
    # cv2::Rect roi_rect = cv2::Rect(bb[0], bb[1], bb[2], bb[3])
    roi = img[bb[0]:bb[1], bb[2]:bb[3]]
    height, width, channel = img.shape
    result_img = np.zeros((height, width, channel), dtype=img.dtype) + 255
    # result_img = np.full((height, width, channel), 255)
    result_img[bb[0]:bb[1], bb[2]:bb[3]] = roi
    return result_img, roi

def cal_filled_img(data_root, resized_data_root, resized_subject_data_root, configuration_root):
    dir_txt = ["train.txt", "val.txt"]
    for i in range(2):
        file_list = format_file_list(data_root, resized_data_root, resized_subject_data_root, configuration_root, dir_txt[i])
        for i in range(len(file_list['segment_file_list'])):
            coord_2d_file = file_list['coord_2d_file_list'][i]
            with open(coord_2d_file, 'rb')as fp:
                coord_2d_val = struct.unpack('d' * 17 * 2, fp.read(17 * 2 * 8))
                coord_2d_val = np.array(coord_2d_val)
                coord_2d_val = np.reshape(coord_2d_val, [17, 2])
            bb = get_bound_box(coord_2d_val)
            images_name = file_list['image_file_list'][i]
            resized_images_name = file_list['resized_image_file_list'][i]
            resized_subject_images_name = file_list['resized_subject_image_file_list'][i]
            img = cv2.imread(images_name)
            filled_img, roi_img = get_filled_img(img, bb)
            filled_img = cv2.resize(filled_img, (image_height, image_width))
            roi_img = cv2.resize(roi_img, (image_height, image_width))
            cv2.imwrite(resized_images_name, filled_img)
            cv2.imwrite(resized_subject_images_name, roi_img)


def cal_depth_init(focal_length, segment, coord_2d):
    '''
    :param focal_length:
    :param segment:
    :param coord_2d:
    :return:
    '''
    segment_3d = np.sqrt(segment)
    upper_joint = [x for x in range(1, 17)]
    lower_joint = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    sement_2d = coord_2d[upper_joint, :] - coord_2d[lower_joint, :]
    sement_2d = sement_2d ** 2
    sement_2d = sement_2d.sum(axis=1)
    sement_2d = np.sqrt(sement_2d)
    depth_init_value = segment_3d / sement_2d
    arg_min = np.argmin(depth_init_value)
    arg_min = np.reshape(arg_min, [-1, 1])
    result = depth_init_value[arg_min]
    depth_init_value = result * focal_length
    return depth_init_value
def cal_init_depth_value():
    file_list = format_file_list("", "", "..\\data_configuration", "train.txt")
    for i in range(len(file_list['segment_file_list'])):
        cam_file = file_list['cam_file_list'][i]
        with open(cam_file, 'rb')as fp:
            cam_val = struct.unpack('d' * 16, fp.read(8 * 16))
        focal_length = cam_val[12:14]  # tf.slice(raw_cam_vec, [12], [2])
        focal_length = (focal_length[0] + focal_length[1])/2.0
        focal_length = np.array(focal_length)
        segment_file = file_list['segment_file_list'][i]
        with open(segment_file, 'rb')as fp:
            segment_val = struct.unpack('d' * 16, fp.read(8 * 16))
            segment_val = np.array(segment_val)
        coord_2d_file = file_list['coord_2d_file_list'][i]
        with open(coord_2d_file, 'rb')as fp:
            coord_2d_val = struct.unpack('d' * 17 * 2, fp.read(17 * 2 * 8))
            coord_2d_val = np.array(coord_2d_val)
            coord_2d_val = np.reshape(coord_2d_val, [17, 2])
        init_depth =cal_depth_init(focal_length, segment_val, coord_2d_val)
        subfolders = cam_file[:-7]
        init_depth_file = subfolders + 'init_depth.bin'
        with open(init_depth_file, 'wb')as fp:
            a = struct.pack('d', init_depth)
            fp.write(a)

    file_list = format_file_list("..\\data_configuration", "val.txt")
    for i in range(len(file_list['segment_file_list'])):
        cam_file = file_list['cam_file_list'][i]
        with open(cam_file, 'rb')as fp:
            cam_val = struct.unpack('d' * 16, fp.read(8 * 16))
        focal_length = cam_val[12:14]  # tf.slice(raw_cam_vec, [12], [2])
        focal_length = (focal_length[0] + focal_length[1]) / 2.0
        focal_length = np.array(focal_length)
        segment_file = file_list['segment_file_list'][i]
        with open(segment_file, 'rb')as fp:
            segment_val = struct.unpack('d' * 16, fp.read(8 * 16))
            segment_val = np.array(segment_val)
        coord_2d_file = file_list['coord_2d_file_list'][i]
        with open(coord_2d_file, 'rb')as fp:
            coord_2d_val = struct.unpack('d' * 17 * 2, fp.read(17 * 2 * 8))
            coord_2d_val = np.array(coord_2d_val)
            coord_2d_val = np.reshape(coord_2d_val, [17, 2])
        init_depth = cal_depth_init(focal_length, segment_val, coord_2d_val)
        subfolders = cam_file[:-7]
        init_depth_file = subfolders + 'init_depth.bin'
        with open(init_depth_file, 'wb')as fp:
            a = struct.pack('d', init_depth)
            fp.write(a)

cal_filled_img("..\\Images-nocontour","..\\filled_resized_images","..\\subject_resized_images","..\\data_configuration")
print("finished!")