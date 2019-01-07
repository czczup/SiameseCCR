import tensorflow as tf
from model import Classifier
import numpy as np
import cv2
import config
import sys
import heapq
import time
import os
import pandas as pd


def test(classifier, sess):
    time1 = time.time()
    max_lists = []
    num = 15020
    for i in range(num):
        name = str(i).zfill(5)
        path = config.train_image_path
        image = cv2.imread(path + "/" + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        image = image[2:46, 2:46].reshape([44, 44, 1]) / 255.0

        prediction = sess.run(classifier.prediction, feed_dict={classifier.X: [image], classifier.training: False})[0]
        max_list = heapq.nlargest(10, range(len(prediction)), prediction.take)
        print(max_list)
        max_lists.append(max_list)
        sys.stdout.write('\r>> Testing image %d/%d' % (i + 1, num))
        sys.stdout.flush()
    time2 = time.time()

    table = pd.read_csv(config.train_labels_path, encoding="gb2312", header=None)
    print(table)
    labels = pd.read_csv("gb2312_level1.csv")
    chinese = [item[2] for item in labels.values]
    id = [item[4] for item in labels.values]
    chinese2id = dict(zip(chinese, id))
    id2chinese = dict(zip(id, chinese))
    results = [chinese2id[item[1]] for item in table.values]

    count_top1 = 0
    count_top5 = 0
    count_top10 = 0
    for index, item in enumerate(results):
        print(id2chinese[item], [id2chinese[item] for item in max_lists[index]])
        if item == max_lists[index][0]:
            count_top1 += 1
        if item in max_lists[index][0:5]:
            count_top5 += 1
        if item in max_lists[index]:
            count_top10 += 1
    print("top1 acc:", count_top1 / num)
    print("top5 acc:", count_top5 / num)
    print("top10 acc:", count_top10 / num)


if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.graph.as_default():
        with sess.as_default():
            classifier = Classifier()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += [var for var in tf.global_variables() if "global_step" in var.name]
            var_list += tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            last_file = tf.train.latest_checkpoint(config.model_save_path)
            print('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
    test(classifier, sess)


