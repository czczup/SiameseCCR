import tensorflow as tf
import os
import random
import sys
import cv2
import numpy as np
import pandas as pd


# 读取汉字对应关系
table = pd.read_csv("../data/gb2312_level1.csv")
value = table.values
ids = [item[3] for item in value]
chars = [item[2].strip() for item in value]
id2char = dict(zip(ids, chars))
char2id = dict(zip(chars, ids))

label = pd.read_csv("../data/train/labels.txt", header=None, encoding="gb2312")
characters = {}
for index, ch in label.values:
    id = char2id[ch]
    filename = str(index).zfill(5)+".jpg"
    if id in characters:
        characters[id].append(filename)
    else:
        characters[id] = [filename]


def get_data(sample_sum):
    data = []
    for i in range(sample_sum):
        item = get_positive_pair()
        data.append(item)
        item = get_negative_pair()
        data.append(item)
        sys.stdout.write('\r>> Loading sample pairs %d/%d' % (i + 1, sample_sum))
        sys.stdout.flush()
    return data


def get_different_randint(start, end):  # 左闭右开
    num1 = np.random.randint(start, end)
    num2 = np.random.randint(start, end)
    while num2 == num1:
        num2 = np.random.randint(start, end)
    return num1, num2


def get_positive_pair():  # 获取正样本对
    id = np.random.randint(0, 3755)  # 随机产生汉字的编号
    index1 = np.random.randint(0, 4)
    index2 = np.random.randint(0, 4)
    path1 = "../data/train/" + characters[id][index1]
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    path2 = "../data/train/" + characters[id][index2]
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    return image1, image2, 1


def get_negative_pair():  # 获取负样本对
    id1, id2 = get_different_randint(0, 3755)  # 随机产生汉字的编号
    index1 = np.random.randint(0, 4)  # 随机产生汉字的编号
    index2 = np.random.randint(0, 4)  # 随机产生汉字的编号
    path1 = "../data/train/" + characters[id1][index1]
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    path2 = "../data/train/" + characters[id2][index2]
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    return image1, image2, 0


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data1, image_data2, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image1': bytes_feature(image_data1),
        'image2': bytes_feature(image_data2),
        'label': int64_feature(class_id),
    }))


def _convert_dataset(data, tfrecord_path, filename):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, filename)
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    length = len(data)
    for index, item in enumerate(data):
        image1 = item[0].tobytes()
        image2 = item[1].tobytes()
        label = item[2]
        example = image_to_tfexample(image1, image2, label)
        tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def generate_train_tfrecord(time, sample_sum):
    data = get_data(sample_sum=sample_sum+10000)
    random.seed(0)
    random.shuffle(data)
    if not os.path.exists("../data/tfrecord/"):
        os.mkdir("../data/tfrecord/")
    _convert_dataset(data[:sample_sum*2], "../data/tfrecord/", "train%d.tfrecord" % time)
    _convert_dataset(data[2*sample_sum:], "../data/tfrecord/", "valid%d.tfrecord" % time)


if __name__ == '__main__':
    generate_train_tfrecord(time=0, sample_sum=10000)

