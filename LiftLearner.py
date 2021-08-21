# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
from data_loader import DataLoader
from inception.inception import *
from vgg import *
from reprojection_image_coord import *
import tensorflow.contrib.slim as slim
import linear_model_utils

linear_model_arg_scope = linear_model_utils.linear_model_arg_scope

class LiftLearner (object):
    def __init__(self, is_training=True):
        self.is_training= is_training
        self.HUMAN_JOINT_SIZE = 17

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
        # if index.dtype != np.dtype('int_'):
        #     raise TypeError("The values of index must be integers")
        data_swaped = np.swapaxes(original, 0, dim)
        index_swaped = np.swapaxes(index, 0, dim)
        gathered = np.choose(index_swaped, data_swaped)
        return np.swapaxes(gathered, 0, dim)


    def cal_depth_init(self, focal_length, segment, coord_2d):
        '''
        :param focal_length: [b, 1]
        :param segment: [b, 16]
        :param coord_2d: [b, 17, 2]
        :return:
        '''
        batch_size, joint_size, _ = coord_2d.get_shape().as_list()
        segment_3d = tf.sqrt(segment)
        upper_joint = [x for x in range(1, batch_size)]
        lower_joint = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        sement_2d = coord_2d[:, upper_joint, :] - coord_2d[:, lower_joint, :]  # [b, 16, 2]
        sement_2d = tf.square(sement_2d) # [b, 16, 2]
        sement_2d = tf.reduce_sum(sement_2d, axis=2)  # [b, 16]
        sement_2d = tf.sqrt(sement_2d)  # [b, 16]
        depth_init_value = tf.divide(segment_3d, sement_2d) # [b, 16]
        arg_max = tf.argmax(depth_init_value, axis=1)  # [b]
        depth_init_value = tf.reshape(depth_init_value, [-1]) # [b*16]
        base = tf.range(batch_size) * (joint_size-1)
        arg_max = arg_max + base
        result = tf.gather(depth_init_value, arg_max)
        result = tf.reshape(result, [-1, joint_size-1])  # [b, 16]
        depth_init_value = result * focal_length
        return depth_init_value

    def kaiming(shape, dtype, partition_info=None):
        """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

        Args
          shape: dimensions of the tf array to initialize
          dtype: data type of the array
          partition_info: (Optional) info about how the variable is partitioned.
            See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
            Needed to be used as an initializer.
        Returns
          Tensorflow array with initial weights
        """
        return (tf.truncated_normal(shape, dtype=dtype) * tf.sqrt(2 / float(shape[0])))


    def two_linear(self, xin, linear_size, idx, residual = False, dropout_keep_prob = 0.8):
        with tf.variable_scope("two_linear_" + str(idx)) as scope:
            first_fc = slim.fully_connected(xin, linear_size,
                                              activation_fn=tf.nn.relu,
                                              scope='First_FC')
            dr_1 = slim.dropout(first_fc, dropout_keep_prob, is_training=self.is_training, scope='Dropout_1')
            second_fc = slim.fully_connected(dr_1, linear_size,
                                            activation_fn=tf.nn.relu,
                                            scope='Second_FC')
            dr_2 = slim.dropout(second_fc, dropout_keep_prob, is_training=self.is_training, scope='Dropout_2')
            # Residual every 2 blocks
            y = (xin + dr_2) if residual else dr_2
        return y

    def line_model(self, xin, linear_size, out_size, residual=False, dropout_keep_prob = 0.8):
        weight_decay = 0.00004  # if don't use L2 regular, set as 1e-10
        with tf.variable_scope("linear_model"):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                weights_initializer=slim.variance_scaling_initializer()):
            # with slim.arg_scope(linear_model_arg_scope()):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=self.is_training):
                    input_fc = slim.fully_connected(xin, linear_size,
                                                    activation_fn=tf.nn.relu,
                                                    scope='Input_FC')
                    y = slim.dropout(input_fc, dropout_keep_prob, is_training=self.is_training, scope='Dropout_Input')

                    for idx in range(2):
                        y = self.two_linear(y, linear_size, idx, residual, dropout_keep_prob)

                    output_fc = slim.fully_connected(y, out_size, activation_fn=None, scope='Output_FC')
        return output_fc


    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.configuration_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.resize_img_height,
                            opt.resize_img_width,
                            )
        with tf.name_scope("data_loading"):
            src_image, focal_length, center_point, segment, coord_3d, coord_2d, init_depth = loader.load_train_batch()  # coord_3d.shape = [b, 17, 3]

            coord_3d_float32 = tf.cast(coord_3d, dtype=tf.float32)
            init_depth_float32 = tf.cast(init_depth, dtype=tf.float32)

            batch_size, joint_size, _ = coord_2d.get_shape().as_list()
            focal_length_float32 = tf.cast(focal_length, dtype=tf.float32)
            focal_length_float32 = tf.reshape(focal_length_float32, [batch_size, -1])
            center_point_float32 = tf.cast(center_point, dtype=tf.float32)
            center_point_float32 = tf.reshape(center_point_float32, [batch_size, -1])
            coord_2d_float32 = tf.cast(coord_2d, dtype=tf.float32)
            coord_2d_float32 = tf.reshape(coord_2d_float32, [batch_size, -1])
            segment_float32 = tf.cast(segment, dtype=tf.float32)
            segment_float32 = tf.sqrt(segment_float32)  # because the read segment is the square of length of segment.
            segment_float32 = tf.reshape(segment_float32, [batch_size, -1])

            src_image = self.preprocess_image(src_image)


        if self.opt.is_v4:
            with slim.arg_scope(inception_v4_arg_scope(use_batch_norm=opt.use_batch_normal_flag)):
                # pred_depth, aux_pred_depth, depth_net_endpoints, pred_ordinal, aux_pred_ordinal, ordinal_net_endpoints = inception_v4(src_image,is_training=self.is_training)
                pred_all_depth, aux_pred_all_depth, all_depth_net_endpoints = inception_v4(
                    src_image, num_classes=17, scope='InceptionV4_Ordinal', is_training=self.is_training)
        else:
            with slim.arg_scope(joint_depth_net_inception_arg_scope()): # inception_v3
                pred_depth, aux_pred_depth, depth_net_endpoints = joint_depth_net_inception(src_image, use_incepiton_FC=opt.use_incepiton_FC, is_training=self.is_training)

        with tf.name_scope("compute_loss"):
            z1_loss = 0
            coord_loss = 0
            ordinal_loss = 0

            z1_mean = tf.constant([opt.z1_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z2_mean = tf.constant([opt.z2_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z3_mean = tf.constant([opt.z3_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z4_mean = tf.constant([opt.z4_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z5_mean = tf.constant([opt.z5_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z6_mean = tf.constant([opt.z6_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z7_mean = tf.constant([opt.z7_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z8_mean = tf.constant([opt.z8_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z9_mean = tf.constant([opt.z9_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z10_mean = tf.constant([opt.z10_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z11_mean = tf.constant([opt.z11_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z12_mean = tf.constant([opt.z12_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z13_mean = tf.constant([opt.z13_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z14_mean = tf.constant([opt.z14_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z15_mean = tf.constant([opt.z15_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z16_mean = tf.constant([opt.z16_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z17_mean = tf.constant([opt.z17_mean] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            all_z_mean = tf.concat(
                [z1_mean, z2_mean, z3_mean, z4_mean, z5_mean, z6_mean, z7_mean, z8_mean, z9_mean, z10_mean, z11_mean,
                 z12_mean, z13_mean, z14_mean, z15_mean, z16_mean, z17_mean], axis=1)  # [b, 17]

            z1_std = tf.constant([opt.z1_std] * opt.batch_size, shape=[opt.batch_size, 1],
                                 dtype=tf.float32)
            z2_std = tf.constant([opt.z2_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z3_std = tf.constant([opt.z3_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z4_std = tf.constant([opt.z4_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z5_std = tf.constant([opt.z5_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z6_std = tf.constant([opt.z6_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z7_std = tf.constant([opt.z7_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z8_std = tf.constant([opt.z8_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z9_std = tf.constant([opt.z9_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z10_std = tf.constant([opt.z10_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z11_std = tf.constant([opt.z11_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z12_std = tf.constant([opt.z12_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z13_std = tf.constant([opt.z13_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z14_std = tf.constant([opt.z14_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z15_std = tf.constant([opt.z15_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z16_std = tf.constant([opt.z16_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            z17_std = tf.constant([opt.z17_std] * opt.batch_size, shape=[opt.batch_size, 1], dtype=tf.float32)
            all_z_std = tf.concat(
                [z1_std, z2_std, z3_std, z4_std, z5_std, z6_std, z7_std, z8_std, z9_std, z10_std, z11_std, z12_std,
                 z13_std, z14_std, z15_std, z16_std, z17_std], axis=1)  # [b, 17]


            all_z = tf.slice(coord_3d_float32, [0, 0, 2], [-1, -1, 1])  # [b, 17, 1]
            all_z = tf.squeeze(all_z, axis=2)  # [b, 17]

            all_z = tf.divide((all_z - all_z_mean), all_z_std)


            pred_all_depth_concat = tf.concat([pred_all_depth, focal_length_float32, center_point_float32, coord_2d_float32, segment_float32], axis=1)  # [b, 55+16]
            pred_line_model = self.line_model(pred_all_depth_concat, opt.linear_size, joint_size, residual = False, dropout_keep_prob = 1.0)

            all_z_loss = tf.abs(tf.subtract(all_z, pred_all_depth)) # L1 loss
            all_z_loss = tf.reduce_mean(all_z_loss)

            aux_all_z_loss = tf.abs(tf.subtract(all_z, aux_pred_all_depth))
            aux_all_z_loss = tf.reduce_mean(aux_all_z_loss)

            line_model_z_loss = tf.abs(tf.subtract(all_z, pred_line_model)) # L1 loss
            line_model_z_loss = tf.reduce_mean(line_model_z_loss)

            all_z_loss = (all_z_loss + aux_all_z_loss + line_model_z_loss) / 3.0

            total_loss = all_z_loss  # + opt.coord_loss_weight * coord_loss

            with tf.name_scope("train_op"):
                train_vars = [var for var in tf.trainable_variables()]

                self.global_step = tf.Variable(0,
                                               name='global_step',
                                               trainable=False)
                self.incr_global_step = tf.assign(self.global_step,
                                                  self.global_step + 1)

                self.learning_rate = tf.Variable(float(opt.learning_rate), trainable=False, dtype=tf.float32,
                                                 name="learning_rate")
                decay_steps = 40000  # empirical
                decay_rate = 0.975  # empirical
                self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps,
                                                                decay_rate)

                optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=opt.beta1, beta2=opt.beta2, epsilon=opt.epsilon, use_locking=False, name='Adam')

                self.train_op = slim.learning.create_train_op(total_loss, optim)

                #  Collect tensors that are useful later (e.g. tf summary)
                self.steps_per_epoch = loader.steps_per_epoch
                self.total_loss = total_loss
                self.coord_loss = coord_loss
                self.ordinal_loss = all_z_loss
                self.z1_loss = z1_loss
    def cal_local_coord_loss(self, coord_3d_groundtruth, recovered_coord_3d, valid_flag):
        """

        :param coord_3d_groundtruth: [b, 17, 3]
        :param recovered_coord_3d:
        :return:
        """
        upper_joint = [x for x in range(1, 17)]
        lower_joint = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        valid_flag = tf.tile(tf.expand_dims(valid_flag, 2), [1, 16, 3])
        for i in range(16):
            recovered_local_coord = recovered_coord_3d[:, upper_joint[i], :] - recovered_coord_3d[:, lower_joint[i], :]
            local_coord = coord_3d_groundtruth[:, upper_joint[i], :] - coord_3d_groundtruth[:, lower_joint[i], :]
            if i ==0:
                stacked_local_coord = tf.expand_dims(local_coord, 1)
                stacked_recovered_local_coord = tf.expand_dims(recovered_local_coord, 1)
            else:
                stacked_local_coord = tf.concat((stacked_local_coord, tf.expand_dims(local_coord, 1)), axis=1)  # P_recovered[:, i + 1, :] = p_recovered
                stacked_recovered_local_coord = tf.concat((stacked_recovered_local_coord, tf.expand_dims(recovered_local_coord, 1)), axis=1)

        local_coord_loss = tf.square(stacked_local_coord - stacked_recovered_local_coord)
        local_coord_loss = tf.sqrt(tf.reduce_sum(local_coord_loss, axis=2))  # [N, 16]
        local_coord_loss = tf.reduce_mean(local_coord_loss)
        return local_coord_loss

    def cal_global_coord_loss(self, coord_3d_groundtruth, recovered_coord_3d, valid_flag):
        coord_loss = tf.square(coord_3d_groundtruth - recovered_coord_3d)
        coord_loss = tf.sqrt(tf.reduce_sum(coord_loss, axis=2))  # [N, 16]
        coord_loss = tf.reduce_mean(coord_loss)
        return coord_loss

    def cal_coord_loss(self, coord_3d_groundtruth, P_image, f, c, L1_square, depth_prediction):

        recovered_coord_3d, valid_flag = back_project_skeleton_to_world(coord_3d_groundtruth, P_image, f, c, L1_square, depth_prediction)

        valid_flag = tf.where(valid_flag > 0.)
        valid_flag = tf.reshape(valid_flag, [1, -1])
        recovered_coord_3d = tf.gather(recovered_coord_3d,valid_flag)
        coord_3d_groundtruth = tf.gather(coord_3d_groundtruth, valid_flag)
        if recovered_coord_3d.shape[0] == 0:
            return 0.

        global_coord_loss = self.cal_global_coord_loss(coord_3d_groundtruth, recovered_coord_3d, valid_flag)
        local_coord_loss = self.cal_local_coord_loss(coord_3d_groundtruth, recovered_coord_3d, valid_flag)
        return  global_coord_loss

    def cal_limb_ordinal_loss(self, z_joints, z_joints_pred):
        """
        :param z_joints: shape =[b, N]
        :param z_joints_pred: shape =[b, N]
        :return:
        """
        limb_upper = z_joints[:, 1:]
        limb_lower = z_joints[:, 0:-1]

        limb_upper_pred = z_joints_pred[:, 1:]
        limb_lower_pred = z_joints_pred[:, 0: -1]

        limb_ordinal_loss = tf.where(tf.greater(limb_upper, limb_lower), \
                                          tf.log(1.0 + tf.exp(limb_lower_pred - limb_upper_pred)), \
                                          tf.log(1.0 + tf.exp(limb_upper_pred - limb_lower_pred)))
        limb_ordinal_loss = tf.reduce_mean(limb_ordinal_loss)
        return limb_ordinal_loss


    def get_equal_threshold(self, joint_num):
        opt = self.opt
        equal_threshold = np.full([opt.batch_size, joint_num], opt.depth_equal_threshold)
        equal_threshold = tf.constant(equal_threshold, dtype=tf.float32)
        return equal_threshold

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) #
        return image * 2. -1.

    def collect_summaries(self):

        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("coord_loss", self.coord_loss)
        tf.summary.scalar("ordinal_loss", self.ordinal_loss)
        tf.summary.scalar("z1_loss", self.z1_loss)

    def load_pretrained_model(self):
        if self.opt.is_v4:
            pretrained_mode = self.opt.v4_pretrained_model_dir
        else:
            pretrained_mode = self.opt.v3_pretrained_model_dir
        print('Loading model check point from {:s}'.format(pretrained_mode))

        if self.opt.is_v4:
            exclusions = ['InceptionV4/Conv2d_1a_3x3/',
                          'InceptionV4/Conv2d_2a_3x3/',
                          'InceptionV4/Conv2d_2b_3x3',
                          'InceptionV4/Mixed_3a',
                          'InceptionV4/Mixed_4a',
                          'InceptionV4/Mixed_5a',
                          'InceptionV4/Mixed_5b',
                          'InceptionV4/Mixed_5c',
                          'InceptionV4/Mixed_5d',
                          'InceptionV4/Mixed_5e',
                          'InceptionV4/Mixed_6e',
                          'InceptionV4/Mixed_6a',
                          'InceptionV4/Mixed_6b',
                          'InceptionV4/Mixed_6c',
                          'InceptionV4/Mixed_6d',
                          'InceptionV4/Mixed_6e',
                          'InceptionV4/Mixed_6f',
                          'InceptionV4/Mixed_6g',
                          'InceptionV4/Mixed_6h',
                          'InceptionV4/Mixed_7a',
                          'InceptionV4/Mixed_7b',
                          'InceptionV4/Mixed_7c',
                          'InceptionV4/Mixed_7d',
                          'InceptionV4/AuxLogits',
                          'InceptionV4/Logits'
                          ]  # inception_v4

        else:
            exclusions = ['InceptionV3/Logits',
                          'InceptionV3/AuxLogits',
                          'InceptionV3/Mixed_7c',
                          'InceptionV3/Mixed_7b',
                          'InceptionV3/Mixed_7a',
                          'InceptionV3/Mixed_6e',
                          'InceptionV3/Mixed_6d',
                          'InceptionV3/Mixed_6c',
                          'InceptionV3/Mixed_6b',
                          'InceptionV3/Mixed_6a'
                          ]  # inception_v3

        inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)
        for var in inception_except_logits:
            print(var.name)
        init_fn = slim.assign_from_checkpoint_fn(pretrained_mode, inception_except_logits,
                                                 ignore_missing_vars=True)
        return init_fn
    def train(self, opt):
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        max_steps = opt.max_epoches * self.steps_per_epoch

        if opt.resume_from_imagenet_model:
            init_fn = self.load_pretrained_model()

            slim.learning.train(train_op=self.train_op, logdir=opt.checkpoint_dir,
                                init_fn=init_fn, number_of_steps=max_steps,
                                save_summaries_secs=20,
                                save_interval_secs=600)
        else:
            with tf.name_scope("parameter_count"):
                parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                                 for v in tf.trainable_variables()])

            self.saver = tf.train.Saver([var for var in slim.get_model_variables()] + \
                                        [self.global_step],
                                        max_to_keep=8)

            sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                     save_summaries_secs=0,
                                     saver=None)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with sv.managed_session(config=config) as sess:
                print('Trainable variables: ')
                for var in slim.get_model_variables(): # for var in tf.trainable_variables():
                    print(var.name)
                print("parameter_count =", sess.run(parameter_count))
                if opt.continue_train:
                    if opt.init_checkpoint_file is None:
                        checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                    else:
                        checkpoint = opt.init_checkpoint_file
                    print("Resume training from previous checkpoint: %s" % checkpoint)
                    self.saver.restore(sess, checkpoint)

                start_time = time.time()
                for step in range(1, max_steps):
                    fetches = {
                        "train": self.train_op,
                        "global_step": self.global_step,
                        "incr_global_step": self.incr_global_step

                    }

                    if step % opt.summary_freq == 0:
                        fetches["loss"] = self.total_loss
                        fetches["summary"] = sv.summary_op

                    results = sess.run(fetches)  #
                    gs = results["global_step"]

                    if step % opt.summary_freq == 0:
                        sv.summary_writer.add_summary(results["summary"], gs)
                        train_epoch = math.ceil(gs / self.steps_per_epoch)
                        train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                        print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.20f" \
                              % (train_epoch, train_step, self.steps_per_epoch, \
                                 (time.time() - start_time) / opt.summary_freq,
                                 results["loss"]))
                        start_time = time.time()

                    if step % opt.save_latest_freq == 0:
                        self.save(sess, opt.checkpoint_dir, gs) # self.save(sess, opt.checkpoint_dir, 'latest')

                    if step % self.steps_per_epoch == 0:
                        self.save(sess, opt.checkpoint_dir, gs)

    def build_test_graph(self, linear_size, use_batch_normal_flag=True):
        input_float32 = tf.placeholder(tf.float32, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        input_norm = self.preprocess_image(input_float32)

        joint_input_float32 = tf.placeholder(tf.float32, [self.batch_size,self.HUMAN_JOINT_SIZE * 2 + 4 + 16], name='joint_input')

        with slim.arg_scope(inception_v4_arg_scope(use_batch_norm=use_batch_normal_flag)):
            if self.two_network:
                pred_depth, aux_pred_depth, depth_net_endpoints = inception_v4(
                    input_norm, num_classes=1, scope='InceptionV4_Depth', is_training=False)
            pred_ordinal, aux_pred_ordinal, ordinal_net_endpoints = inception_v4(
                input_norm, num_classes=17, scope='InceptionV4_Ordinal', is_training=False)

        pred_all_depth_concat = tf.concat([pred_ordinal, joint_input_float32], axis=1)  # [b, 55]
        pred_line_model = self.line_model(pred_all_depth_concat, linear_size, self.HUMAN_JOINT_SIZE,
                                          dropout_keep_prob=1.0)

        self.inputs = input_float32
        self.joint_inputs = joint_input_float32

        self.pred_ordinal = pred_line_model
        self.aux_pred_ordinal = aux_pred_ordinal
        self.ordinal_net_endpoints = ordinal_net_endpoints

    def setup_inference(self,
                        img_height,
                        img_width,
                        linear_size,
                        batch_size=1,
                        use_batch_normal_flag=True,
                        two_network=False):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.two_network = two_network
        self.build_test_graph(linear_size, use_batch_normal_flag)

    def inference(self, inputs, joint_inputs, sess):
        fetches = {}
        fetches['ordinal'] = self.pred_ordinal
        results = sess.run(fetches, feed_dict={self.inputs:inputs, self.joint_inputs:joint_inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
