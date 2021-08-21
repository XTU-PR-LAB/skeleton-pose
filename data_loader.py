# -*- coding: utf-8 -*-
from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np
# import cupy as np
# import minpy.numpy as np
import struct
import cv2
import PIL.Image as pil

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 configuration_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 resize_img_height = None,
                 resize_img_width = None,
                 use_GT_flag = None
                 ):
        self.dataset_dir = dataset_dir
        self.configuration_dir = configuration_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.resize_img_height = resize_img_height
        self.resize_img_width = resize_img_width
        self.use_GT_flag = use_GT_flag
        self.read_mean_image(os.path.join(configuration_dir, 'mean_image.bin'))
        self.act_num = 15

    def read_mean_image(self, file_name):
        image_mean = np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        with open(file_name, "rb") as binf:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    # for k in range(3):
                    temp = binf.read(4 * 3)
                    data_raw = struct.unpack('f' * 3, temp)
                    data_raw = np.reshape(np.array(data_raw), [1, -1])
                    image_mean[i, j, :] = data_raw
        image_mean[:, :,[0,1,2]] = image_mean[:, :,[2,1,0]]
        self.image_mean_np = image_mean
        image_mean = tf.constant(image_mean, dtype=tf.float32)
        image_mean = tf.expand_dims(image_mean,axis=0)
        image_mean = tf.tile(image_mean, [self.batch_size, 1, 1, 1])
        self.image_mean = image_mean


    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, self.configuration_dir, "train.txt")
        init_depth_paths_queue = tf.train.string_input_producer(
            file_list['init_depth_file_list'],
            seed=seed,
            shuffle=True)
        image_paths_queue = tf.train.string_input_producer(  #
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)
        cam_paths_queue = tf.train.string_input_producer(  #
            file_list['cam_file_list'],
            seed=seed,
            shuffle=True)
        coord_3d_paths_queue = tf.train.string_input_producer(  #
            file_list['coord_3d_file_list'],
            seed=seed,
            shuffle=True)
        coord_2d_paths_queue = tf.train.string_input_producer(  #
            file_list['coord_2d_file_list'],
            seed=seed,
            shuffle=True)
        segment_paths_queue = tf.train.string_input_producer(  #
            file_list['segment_file_list'],
            seed=seed,
            shuffle=True)

        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)
        # Load ini depth
        record_bytes = 8
        init_depth_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, init_depth_contents = init_depth_reader.read(init_depth_paths_queue)
        init_depth = tf.decode_raw(init_depth_contents, tf.float64)
        init_depth = tf.reshape(init_depth, [1])

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue) #
        image_seq = tf.image.decode_jpeg(image_contents, channels=3,dct_method='INTEGER_ACCURATE')
        src_image = tf.slice(image_seq,
                               [0, 0, 0],
                               [self.img_height, self.img_width, 3])

        # Load camera intrinsics
        record_bytes = 16 * 8
        cam_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, cam_contents = cam_reader.read(cam_paths_queue)
        cam_vec = tf.decode_raw(cam_contents, tf.float64)
        raw_cam_vec = tf.reshape(cam_vec, [16])
        focal_length = tf.slice(raw_cam_vec, [12], [2])
        focal_length = tf.reshape(focal_length, [2])

        center_point = tf.slice(raw_cam_vec, [14], [2])
        center_point = tf.reshape(center_point, [2])

        # Load segment
        record_bytes = 16 * 8
        segment_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, segment_contents = segment_reader.read(segment_paths_queue)
        segment_vec = tf.decode_raw(segment_contents, tf.float64)
        segment = tf.reshape(segment_vec, [16])

        # Load 3d coord
        record_bytes = 17 * 3 * 8
        coord_3d_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, coord_3d_contents = coord_3d_reader.read(coord_3d_paths_queue)
        coord_3d_vec = tf.decode_raw(coord_3d_contents, tf.float64)
        coord_3d = tf.reshape(coord_3d_vec, [17, 3])

        # Load 2d image coord
        record_bytes = 17 * 2 * 8
        coord_2d_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, coord_2d_contents = coord_2d_reader.read(coord_2d_paths_queue)
        coord_2d_vec = tf.decode_raw(coord_2d_contents, tf.float64)
        coord_2d = tf.reshape(coord_2d_vec, [17, 2])

        # Form training batches
        src_image, focal_length, center_point, segment, coord_3d, coord_2d, init_depth= \
                tf.train.batch([src_image, focal_length, center_point, segment, coord_3d, coord_2d, init_depth],
                               batch_size=self.batch_size)

        if(self.img_height > self.resize_img_height or self.img_width > self.resize_img_width):
            src_image = tf.image.resize_area(src_image, [self.resize_img_height, self.resize_img_width])
        src_image = tf.cast(src_image, dtype=tf.float32)
        return src_image, focal_length, center_point, segment, coord_3d, coord_2d, init_depth

    #
    def read_file_list(self, data_root, configuration_root,file_name):
        with open(os.path.join(configuration_root, file_name)) as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        frame_ids = [int(i) for i in frame_ids]
        cam_file_list = []
        coord_3d_file_list = []
        predicted_3d_file_list = []
        estimated_3d_file_list = []
        image_file_list = []
        viz_image_file_list = []
        coord_2d_file_list = []
        for i in range(len(frame_ids)):
            cam_file_list.append([os.path.join(configuration_root, subfolders[i],
                                               subfolders[i] + "_" + "{0:0>6d}".format(j) + "_cam.bin") for j in
                                  range(1, 1 + frame_ids[i])])

            coord_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                    subfolders[i] + "_" + "{0:0>6d}".format(j) + "_pose3d_cam.bin")
                                       for j in range(1, 1 + frame_ids[i])])

            predicted_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                    subfolders[i] + "_" + "{0:0>6d}".format(j) + "_predicted_pose3d.bin")
                                       for j in range(1, 1 + frame_ids[i])])

            estimated_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                        subfolders[i] + "_" + "{0:0>6d}".format(
                                                            j) + "_estimated_pose3d.bin")
                                           for j in range(1, 1 + frame_ids[i])])

            image_file_list.append([os.path.join(data_root, subfolders[i],
                                                   subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                      range(1, 1 + frame_ids[i])])
            viz_image_file_list.append([os.path.join("..\\visualize_images", subfolders[i],
                                                     subfolders[i] + "_" + "{0:0>6d}".format(j) + ".jpg") for j in
                                        range(1, 1 + frame_ids[i])])
            coord_2d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                    subfolders[i] + "_" + "{0:0>6d}".format(j) + "_image_coord.bin")
                                       for j in
                                       range(1, 1 + frame_ids[i])])
        cam_file_list = [i for item in cam_file_list for i in item]
        coord_3d_file_list = [i for item in coord_3d_file_list for i in item]
        predicted_3d_file_list = [i for item in predicted_3d_file_list for i in item]
        estimated_3d_file_list = [i for item in estimated_3d_file_list for i in item]
        image_file_list = [i for item in image_file_list for i in item]
        viz_image_file_list = [i for item in viz_image_file_list for i in item]
        coord_2d_file_list = [i for item in coord_2d_file_list for i in item]
        all_list = {}
        all_list['cam_file_list'] = cam_file_list
        all_list['coord_3d_file_list'] = coord_3d_file_list
        all_list['image_file_list'] = image_file_list
        all_list['viz_image_file_list'] = viz_image_file_list
        all_list['predicted_3d_file_list'] = predicted_3d_file_list
        all_list['estimated_3d_file_list'] = estimated_3d_file_list
        all_list['coord_2d_file_list'] = coord_2d_file_list
        return all_list
    def format_file_list(self, data_root, configuration_root,file_name):
        with open(os.path.join(configuration_root, file_name)) as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        frame_ids = [int(i) for i in frame_ids]
        init_depth_file_list = []
        cam_file_list = []
        coord_3d_file_list = []
        predicted_z1_file_list = []
        predicted_3d_file_list = []
        recovered_3d_file_list = []
        coord_2d_file_list = []
        segment_file_list = []
        image_file_list = []
        for i in range(len(frame_ids)):

            init_depth_file_list.append([os.path.join(configuration_root, subfolders[i],
                                               subfolders[i] + "_" + "{0:0>6d}".format(j) + "_init_depth.bin") for j in
                                  range(1, 1 + frame_ids[i])])
            cam_file_list.append([os.path.join(configuration_root, subfolders[i],
                                               subfolders[i] + "_" + "{0:0>6d}".format(j) + "_cam.bin") for j in
                                  range(1, 1 + frame_ids[i])])

            coord_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                    subfolders[i] + "_" + "{0:0>6d}".format(j) + "_pose3d_cam.bin")
                                       for j in range(1, 1 + frame_ids[i])])
            predicted_z1_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                        subfolders[i] + "_" + "{0:0>6d}".format(
                                                            j) + "_predicted_z1.bin")
                                           for j in range(1, 1 + frame_ids[i])])
            predicted_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                        subfolders[i] + "_" + "{0:0>6d}".format(
                                                            j) + "_predicted_pose3d.bin")
                                           for j in range(1, 1 + frame_ids[i])])
            recovered_3d_file_list.append([os.path.join(configuration_root, subfolders[i],
                                                    subfolders[i] + "_" + "{0:0>6d}".format(j) + "_predicted_pose3d_cam.bin")
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
        init_depth_file_list = [i for item in init_depth_file_list for i in item]
        cam_file_list = [i for item in cam_file_list for i in item]
        coord_3d_file_list = [i for item in coord_3d_file_list for i in item]
        predicted_z1_file_list = [i for item in predicted_z1_file_list for i in item]
        predicted_3d_file_list = [i for item in predicted_3d_file_list for i in item]
        recovered_3d_file_list = [i for item in recovered_3d_file_list for i in item]
        coord_2d_file_list = [i for item in coord_2d_file_list for i in item]
        segment_file_list = [i for item in segment_file_list for i in item]
        image_file_list = [i for item in image_file_list for i in item]
        all_list = {}
        all_list['init_depth_file_list'] = init_depth_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['coord_3d_file_list'] = coord_3d_file_list
        all_list['recovered_3d_file_list'] = recovered_3d_file_list
        all_list['coord_2d_file_list'] = coord_2d_file_list
        all_list['segment_file_list'] = segment_file_list
        all_list['image_file_list'] = image_file_list
        all_list['predicted_z1_file_list'] = predicted_z1_file_list
        all_list['predicted_3d_file_list'] = predicted_3d_file_list
        return all_list

    def read_cam_file(self, file_name):
        with open(file_name, "rb") as binf:
            temp = binf.read(8 * 16)
            data_raw = struct.unpack('d' * 16, temp)
            data_raw = np.array(data_raw)
            f = data_raw[12:14]
            f = np.reshape(f, [2])
            f = f.tolist()
            c = data_raw[14:]
            c = np.reshape(c, [2])
            c = c.tolist()
        return f, c
    def read_predicted_z1_file(self, file_name):
        with open(file_name, "rb") as f:
            temp = f.read(8)
            data_raw = struct.unpack('d', temp)
            data_raw = np.reshape(np.array(data_raw), [1]).tolist()
        return data_raw
    def read_all_cam_file(self, file_name):
        with open(file_name, "rb") as binf:
            temp = binf.read(8 * 16)
            data_raw = struct.unpack('d' * 16, temp)
            data_raw = np.array(data_raw)
            R = data_raw[:9]
            R = np.reshape(R, [3,3])
            t = data_raw[9:12]
            t = np.reshape(t, [1, 3])
            f = data_raw[12:14]
            f = np.reshape(f, [2])
            # f = f.tolist()
            c = data_raw[14:]
            c = np.reshape(c, [2])
            # c = c.tolist()
        return R, t, f, c

    def read_seg_file(self, file_name):
        with open(file_name, "rb") as f:
            temp = f.read(8 * 16)
            data_raw = struct.unpack('d' * 16, temp)
            data_raw = np.reshape(np.array(data_raw), [16]).tolist()

        return data_raw

    def read_predicted_coord_3d_file(self,file_name):
        with open(file_name, 'rb')as f:
            temp = f.read(8 * 16 * 3)
            data_raw = struct.unpack('d' * 16 * 3, temp)
            data_raw = np.reshape(np.array(data_raw), [-1, 3]).tolist()
        return data_raw

    def read_coord_3d_file(self,file_name):
        with open(file_name, 'rb')as f:
            temp = f.read(8 * 17 * 3)
            data_raw = struct.unpack('d' * 17 * 3, temp)
            data_raw = np.reshape(np.array(data_raw), [-1, 3]).tolist()
        return data_raw

    def read_coord_2d_file(self,file_name):
        with open(file_name, 'rb')as f:
            temp = f.read(17 * 2 * 8)
            data_raw = struct.unpack('d' * 17 * 2, temp)
            data_raw = np.reshape(np.array(data_raw), [-1, 2]).tolist()
        return data_raw

    def read_image_file(self, file_name):

        src_image = cv2.imread(file_name)

        if (self.img_height > self.resize_img_height or self.img_width > self.resize_img_width):
            src_image = cv2.resize(src_image, (self.resize_img_height, self.resize_img_width))

        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        src_image = src_image.tolist()
        return src_image

    def read_init_depth_file(self, file_name):
        with open(file_name, 'rb')as f:
            temp = f.read(8)
            data_raw = struct.unpack('d', temp)
            data_raw = np.reshape(np.array(data_raw), [1]).tolist()
        return data_raw

    def init_test_batch(self, dir_name, action_number):

        self.file_list = []
        self.batch_no = []
        self.sample_num = []
        self.max_batch = []
        for i in range(action_number):
            self.file_list.append(self.format_file_list(self.dataset_dir, self.configuration_dir, dir_name[i]))
            self.batch_no.append(0)
            self.sample_num.append(len(self.file_list[i]['cam_file_list']))
            self.max_batch.append(self.sample_num[i] // self.batch_size)

    def get_next_batch(self, act_no):
        input_imag = []
        P = []
        P_image = []
        f = []
        c = []
        L1_square = []
        recovered_P = []
        subject_depth = []
        predicted_z1 = []
        predicted_3d_file_batch = []

        if self.batch_no[act_no] < self.max_batch[act_no]:
            for i in range(self.batch_size):
                index = i + self.batch_no[act_no] * self.batch_size
                predicted_3d_file_batch.append(self.file_list[act_no]['predicted_3d_file_list'][index])
                z1 = self.read_predicted_z1_file(self.file_list[act_no]['predicted_z1_file_list'][index])
                predicted_z1.append(z1)
                focal, cc = self.read_cam_file(self.file_list[act_no]['cam_file_list'][index])
                f.append(focal)
                c.append(cc)
                seg_len = self.read_seg_file(self.file_list[act_no]['segment_file_list'][index])
                L1_square.append(seg_len)
                coord_3d = self.read_coord_3d_file(self.file_list[act_no]['coord_3d_file_list'][index])
                P.append(coord_3d)
                coord_2d = self.read_coord_2d_file(self.file_list[act_no]['coord_2d_file_list'][index])
                P_image.append(coord_2d)

                src_img = self.read_image_file(self.file_list[act_no]['image_file_list'][index])
                input_imag.append(src_img)

                if self.use_GT_flag == False:
                    recovered_coord_3d = self.read_coord_3d_file(self.file_list[act_no]['recovered_3d_file_list'][index])
                    recovered_P.append(recovered_coord_3d)

        self.batch_no[act_no] += 1
        return np.array(input_imag), np.array(P), np.array(P_image), np.array(f), np.array(c), np.array(L1_square), np.array(recovered_P),  np.array(predicted_z1), predicted_3d_file_batch




