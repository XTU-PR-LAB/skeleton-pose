# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from LiftLearner import LiftLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "..\\Human3.6M\\filled_resized_images", "Dataset directory")
flags.DEFINE_string("configuration_dir", "..\\data_configuration", "Configurate file directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_string("v3_pretrained_model_dir", "..\\inception_v3_2016_08_28\\inception_v3.ckpt", "Trained inception_v3 model")
flags.DEFINE_string("v4_pretrained_model_dir", "..\\inception_v4_2016_09_09\\inception_v4.ckpt", "Trained inception_v4 model")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("beta2", 0.999, "Momentum2 term of adam")
flags.DEFINE_float("epsilon", 0.00000001, "epsilon term of adam")

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
flags.DEFINE_float("z17_mean", 5110.18772794496, "the mean of z1 of samples")

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


flags.DEFINE_float("z1_loss_weight", 1.0, "Weight for depth of the root joint")
flags.DEFINE_float("coord_loss_weight", 1.0, "Weight for coord of each joints")
flags.DEFINE_float("ordinal_loss_weight", 1.0, "Weight for fundmental matrix")
flags.DEFINE_integer("batch_size", 40, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 299, "Image height")
flags.DEFINE_integer("img_width", 299, "Image width")
flags.DEFINE_integer("resize_img_height", 299, "Image height")
flags.DEFINE_integer("resize_img_width", 299, "Image width")
flags.DEFINE_float("depth_equal_threshold", 0.01, "The threshold to indicate whether the depthes of joints are equal")  #
flags.DEFINE_integer("max_epoches", 300, "Maximum number of training epoches")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 100000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", True, "Continue training from previous checkpoint")
flags.DEFINE_boolean("use_incepiton_FC", True, "Use the original fully connection layer in inception_v3")
flags.DEFINE_boolean("resume_from_imagenet_model", False, "Resume the weight from the model which trained on ImageNet")
flags.DEFINE_boolean("is_v4", True, "Indicate the net is inception_v3 or inception_v4")
flags.DEFINE_boolean("use_batch_normal_flag", True, "Indicate the net use batch normal")
flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    lift_learner = LiftLearner(is_training=True)
    lift_learner.train(FLAGS)
    # sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
