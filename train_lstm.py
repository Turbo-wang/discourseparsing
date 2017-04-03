import tensorflow as tf
import sys
import numpy as np
import types
import os
import time
import datetime
from SimpleLSTM import SimpleLSTM
import sklearn as sk
# from data_xuejie import WE_process
# from data_xuejie import embedding_process
from data_utils import DataUtil

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def train(x1_inputs_batch, x2_inputs_batch, y):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            lstm = SimpleLSTM(50)
            sess.run(tf.initialize_all_variables())
            def train_step(x1_inputs_batch, x2_inputs_batch, y_inputs_batch):
                lstm.x1_inputs = x1_inputs_batch
                lstm.x2_inputs = x2_inputs_batch
                lstm.y = y_inputs_batch

                _, loss = sess.run(lstm.optimizer, lstm.loss)
                return loss

            for epo in range(1, epoch_num+1):
                print("epo : %s " % epo)
                avg_loss = 0.
                avg_accuracy = 0.
                for x1_inputs, x2_inputs, y in zip(x1_inputs_batch, x2_inputs_batch, y_inputs_batch):
                    # X_indexs_batch = X_indexs_batch_list[i]
                    # X_others_batch = X_others_batch_list[i]
                    # Y_batch = Y_batch_list[i]
                    loss = train_step(x1_inputs, x2_inputs, y)

                    avg_loss += float(loss) / len(y_inputs_batch)
                    # avg_accuracy += float(accuracy) / len(y_inputs_batch)
                print("avg_loss: ", avg_loss, " avg_accuracy: ", avg_accuracy)

if __name__ == '__main__':
    # train_file = "one_vs_others_standard/train_sec_02_20_EntRel_to_Expansion.json"
    # WE,pos_WE=WE_process()
    # X_train_1,X_train_2,X_train_pos_1,X_train_pos_2,y_train=embedding_process(train_file)
    # print X_train_1[0]
    d = DataUtil()
    d.get_sentence_pairs("dev_sec_00_01.json", "one_vs_others_standard/Comparison_vs_others")
    x1_inputs, x2_inputs, y_inputs = d.get_train_data()
    batch_size = 128
    x1_inputs_batch, x2_inputs_batch, y_inputs_batch = d.construct_batch_matrix(x1_inputs, x2_inputs, y_inputs, batch_size)
    print x1_inputs_batch[0][0]