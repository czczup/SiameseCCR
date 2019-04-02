from generate_train_tfrecord import generate_train_tfrecord
from reconstruct_train_tfrecord import reconstruct_train_tfrecord
from train import train
from test import test
import tensorflow as tf
from model import Siamese
import os


def init_model(class_num):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.graph.as_default():
        with sess.as_default():
            siamese = Siamese(class_num=class_num)
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += [var for var in tf.global_variables() if "global_step" in var.name]
            var_list += tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
            last_file = tf.train.latest_checkpoint("../model/")
            if last_file:
                print('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)
    return sess, saver, siamese


def main():
    train_time = 0
    sample_sum = 300000
    if not os.path.exists("../data/tfrecord/train0.tfrecord"):
        generate_train_tfrecord(train_time, sample_sum=sample_sum)
    while True:
        print(train_time)
        sess, saver, siamese = init_model(class_num=3755)
        if not os.path.exists("../result/log/train%d.log" % train_time):
            train(sess, saver, siamese, train_time, data_sum=sample_sum*2, epoch=10)
        if not os.path.exists("../result/result-train%d.txt" % train_time):
            test(siamese, sess, dataset="train", train_time=train_time)
        if not os.path.exists("../result/result-test-seen%d.txt" % train_time):
            test(siamese, sess, dataset="test-seen", train_time=train_time)
        tf.reset_default_graph()
        sess, saver, siamese = init_model(class_num=3008)
        if not os.path.exists("../result/result-test-unseen%d.txt" % train_time):
            test(siamese, sess, dataset="test-unseen", train_time=train_time)
        tf.reset_default_graph()
        sess, saver, siamese = init_model(class_num=3008+3755)
        if not os.path.exists("../result/result-test%d.txt" % train_time):
            test(siamese, sess, dataset="test", train_time=train_time)  # 用测试集测试
        if not os.path.exists("../data/tfrecord/train%d.tfrecord" % (train_time+1)):
            reconstruct_train_tfrecord(train_time, sample_sum=sample_sum)  # 重构训练集
        tf.reset_default_graph()  # 清空计算图
        train_time += 1


if __name__ == '__main__':
    deviceId = input("please input device id (0-7): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    main()
