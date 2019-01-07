import tensorflow as tf
from model import Siamese
import numpy as np
import cv2
import config
import sys
import heapq
import time
import os
import pandas as pd


def test_character(siamese, sess, part=None):
    class Character:
        def __init__(self, image, label, feature=None):
            self.image = image.reshape([44, 44, 1]) / 255.0
            self.label = label
            self.feature = feature

    if part=="seen":
        path = config.seen_template_path+"/"
    elif part=="unseen":
        path = config.unseen_template_path+"/"
    template_list = []
    for file in os.listdir(path):
        image = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE)
        image = image[2:46, 2:46]
        label = file[0]
        feature = sess.run(siamese.right_output, feed_dict={siamese.right: [image.reshape([44, 44, 1]) / 255.0],
                                                                  siamese.training: False})[0]
        template = Character(image, label, feature)
        template_list.append(template)
    print("字符模板加载完成")
    print("字符模板总数为%d"%len(template_list))

    if part=="seen":
        table = pd.read_csv(config.test_seen_labels_path, header=None, encoding='gb2312')
    elif part=="unseen":
        table = pd.read_csv(config.test_unseen_labels_path, header=None, encoding='gb2312')
    labels = [item[1] for item in table.values]

    time1 = time.time()

    character_list = []
    for i in range(10000):
        name = str(i).zfill(4)
        if part == "seen":
            path = config.test_seen_image_path
        elif part == "unseen":
            path = config.test_unseen_image_path
        image = cv2.imread(path + "/" + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        image = image[2:46, 2:46]
        character = Character(image, labels[i])
        character_list.append(character)

    pred_characters = []
    true_characters = []
    for index, character in enumerate(character_list):
        feature = sess.run(siamese.left_output, feed_dict={siamese.left: [character.image], siamese.training: False})
        prediction = sess.run(y_, feed_dict={image_feature: np.tile(feature, (len(template_list), 1)),
                                               template_feature: [template.feature for template in template_list],
                                               siamese.training: False})
        min_list = heapq.nsmallest(10, range(len(prediction)), prediction.take)
        pred_character = [template_list[item].label for item in min_list]
        pred_characters.append(pred_character)
        true_characters.append(character.label)

        sys.stdout.write('\r>> Testing image %d/%d' % (index + 1, 10000))
        sys.stdout.flush()
    time2 = time.time()
    print("\nUsing time:", "%.2f" % (time2 - time1) + "s")

    count_top1 = 0
    count_top5 = 0
    count_top10 = 0
    for index, item in enumerate(true_characters):
        if item == pred_characters[index][0]:
            count_top1 += 1
        if item in pred_characters[index][0:5]:
            count_top5 += 1
        if item in pred_characters[index]:
            count_top10 += 1
    print("top1 acc:", count_top1 / len(true_characters))
    print("top5 acc:", count_top5 / len(true_characters))
    print("top10 acc:", count_top10 / len(true_characters))


def test_seen_and_unseen(siamese, sess):
    class Character:
        def __init__(self, image, label, feature=None):
            self.image = image.reshape([44, 44, 1]) / 255.0
            self.label = label
            self.feature = feature

    template_list = []
    for path in [config.seen_template_path+"/", config.unseen_template_path+"/"]:
        for file in os.listdir(path):
            image = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE)
            image = image[2:46, 2:46]
            label = file[0]
            feature = sess.run(siamese.right_output, feed_dict={siamese.right: [image.reshape([44, 44, 1]) / 255.0],
                                                                      siamese.training: False})[0]
            template = Character(image, label, feature)
            template_list.append(template)
        print("字符模板加载完成")
        print("字符模板总数为%d"%len(template_list))

    table1 = pd.read_csv(config.test_seen_labels_path, header=None, encoding='gb2312')
    table2 = pd.read_csv(config.test_unseen_labels_path, header=None, encoding='gb2312')
    labels = [item[1] for item in table1.values] + [item[1] for item in table2.values]

    time1 = time.time()

    character_list = []
    cnt = 0
    for path in [config.test_seen_image_path, config.test_unseen_image_path]:
        for i in range(10000):
            name = str(i).zfill(4)
            image = cv2.imread(path + "/" + name + ".jpg", cv2.IMREAD_GRAYSCALE)
            image = image[2:46, 2:46]
            character = Character(image, labels[cnt])
            character_list.append(character)
            cnt += 1

    pred_characters = []
    true_characters = []
    for index, character in enumerate(character_list):
        feature = sess.run(siamese.left_output, feed_dict={siamese.left: [character.image], siamese.training: False})
        prediction = sess.run(y_, feed_dict={image_feature: np.tile(feature, (len(template_list), 1)),
                                               template_feature: [template.feature for template in template_list],
                                               siamese.training: False})
        min_list = heapq.nsmallest(10, range(len(prediction)), prediction.take)
        pred_character = [template_list[item].label for item in min_list]
        pred_characters.append(pred_character)
        true_characters.append(character.label)

        sys.stdout.write('\r>> Testing image %d/%d' % (index + 1, 10000))
        sys.stdout.flush()
    time2 = time.time()
    print("\nUsing time:", "%.2f" % (time2 - time1) + "s")

    count_top1 = 0
    count_top5 = 0
    count_top10 = 0
    for index, item in enumerate(true_characters):
        if item == pred_characters[index][0]:
            count_top1 += 1
        if item in pred_characters[index][0:5]:
            count_top5 += 1
        if item in pred_characters[index]:
            count_top10 += 1
    print("top1 acc:", count_top1 / len(true_characters))
    print("top5 acc:", count_top5 / len(true_characters))
    print("top10 acc:", count_top10 / len(true_characters))


def test_training_set(siamese, sess):
    class Character:
        def __init__(self, image, label, feature=None):
            self.image = image.reshape([44, 44, 1]) / 255.0
            self.label = label
            self.feature = feature

    path = config.seen_template_path+"/"
    template_list = []
    for file in os.listdir(path):
        image = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE)
        image = image[2:46, 2:46]
        label = file[0]
        feature = sess.run(siamese.right_output, feed_dict={siamese.right: [image.reshape([44, 44, 1]) / 255.0],
                                                                  siamese.training: False})[0]
        template = Character(image, label, feature)
        template_list.append(template)
    print("字符模板加载完成")
    print("字符模板总数为%d"%len(template_list))


    table = pd.read_csv(config.train_labels_path, header=None, encoding='gb2312')
    labels = [item[1] for item in table.values]

    time1 = time.time()

    character_list = []
    for i in range(15020):
        name = str(i).zfill(5)
        path = config.train_image_path
        image = cv2.imread(path + "/" + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        image = image[2:46, 2:46]
        character = Character(image, labels[i])
        character_list.append(character)

    pred_characters = []
    true_characters = []
    for index, character in enumerate(character_list):
        feature = sess.run(siamese.left_output, feed_dict={siamese.left: [character.image], siamese.training: False})
        prediction = sess.run(y_, feed_dict={image_feature: np.tile(feature, (len(template_list), 1)),
                                               template_feature: [template.feature for template in template_list],
                                               siamese.training: False})
        min_list = heapq.nsmallest(11, range(len(prediction)), prediction.take)
        pred_character = [template_list[item].label for item in min_list]
        pred_characters.append(pred_character)
        true_characters.append(character.label)

        sys.stdout.write('\r>> Testing image %d/%d' % (index + 1, 10000))
        sys.stdout.flush()
    time2 = time.time()
    print("\nUsing time:", "%.2f" % (time2 - time1) + "s")

    count_top1 = 0
    count_top5 = 0
    count_top10 = 0
    f = open("results.txt", "w+")
    for index, item in enumerate(true_characters):
        temp = list(pred_characters[index])
        if item in temp:
            temp.remove(item)
        print(item, pred_characters[index])
        f.write(item+","+"".join(temp[0:10])+"\n")
        if item == pred_characters[index][0]:
            count_top1 += 1
        if item in pred_characters[index][0:5]:
            count_top5 += 1
        if item in pred_characters[index][0:10]:
            count_top10 += 1
    print("top1 acc:", count_top1 / len(true_characters))
    print("top5 acc:", count_top5 / len(true_characters))
    print("top10 acc:", count_top10 / len(true_characters))


if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.graph.as_default():
        with sess.as_default():
            siamese = Siamese()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += [var for var in tf.global_variables() if "global_step" in var.name]
            var_list += tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            last_file = tf.train.latest_checkpoint(config.model_save_path)
            # var_list = [var for var in tf.global_variables() if "global_step" in var.name]+tf.trainable_variables()
            print('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)

            template_feature = tf.placeholder(tf.float32, [None, 256], name="template_feature")
            image_feature = tf.placeholder(tf.float32, [None, 256], name="image_feature")
            # image_feature = tf.tile(siamese.left_output, multiples=[3755, 1])
            output_difference = tf.abs(image_feature-template_feature)
            wx_plus_b = tf.matmul(output_difference, siamese.test_param[0])+siamese.test_param[1]
            y_ = tf.nn.sigmoid(wx_plus_b, name='distance')

    test_training_set(siamese, sess)
    test_character(siamese, sess, part="seen")
    test_character(siamese, sess, part="unseen")
    test_seen_and_unseen(siamese, sess)

