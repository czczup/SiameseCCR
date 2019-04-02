import time
import tensorflow as tf
import os
import pandas as pd


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image1': tf.FixedLenFeature([], tf.string),
                                           'image2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image1 = tf.reshape(image1, [48, 48, 1])
    image1 = tf.random_crop(image1, [44, 44, 1])
    image1 = tf.cast(image1, tf.float32) / 255.0

    image2 = tf.decode_raw(features['image2'], tf.uint8)
    image2 = tf.reshape(image2, [48, 48, 1])
    image2 = tf.random_crop(image2, [44, 44, 1])
    image2 = tf.cast(image2, tf.float32) / 255.0

    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    label = tf.reshape(label, [1])
    return image1, image2, label


def load_training_set(train_time):
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode("../data/tfrecord/train%d.tfrecord" % train_time)
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=5120, min_after_dequeue=5000, num_threads=16
        )
    return image_batch_train1, image_batch_train2, label_batch_train


def load_valid_set(train_time):
    with tf.name_scope('input_valid'):
        image_valid1, image_valid2, label_valid = read_and_decode("../data/tfrecord/valid%d.tfrecord" % train_time)
        image_batch_valid1, image_batch_valid2, label_batch_valid = tf.train.shuffle_batch(
            [image_valid1, image_valid2, label_valid], batch_size=512, capacity=5120, min_after_dequeue=5000, num_threads=16
        )
    return image_batch_valid1, image_batch_valid2, label_batch_valid


def read_mapping():
    table = pd.read_csv("../data/gb2312_level1.csv")
    value = table.values
    ids = [item[3] for item in value]
    chars = [item[2].strip() for item in value]
    id2char = dict(zip(ids, chars))
    char2id = dict(zip(chars, ids))
    return id2char, char2id


def train(sess, saver, siamese, train_time, data_sum, epoch):
    step_ = sess.run(siamese.global_step)

    if train_time > 0:
        step_ = step_-train_time*epoch*(data_sum//siamese.batch_size)

    epoch_start = step_ // (data_sum//siamese.batch_size)
    step_start = step_ % (data_sum//siamese.batch_size)
    image_batch_train1, image_batch_train2, label_batch_train = load_training_set(train_time)
    image_batch_valid1, image_batch_valid2, label_batch_valid = load_valid_set(train_time)

    writer_train = tf.summary.FileWriter("../model/log/train", sess.graph)
    writer_valid = tf.summary.FileWriter("../model/log/valid", sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(epoch_start, epoch):
        for step in range(step_start, data_sum//siamese.batch_size):
            time1 = time.time()
            image_train1, image_train2, label_train, step_ = sess.run(
                [image_batch_train1, image_batch_train2, label_batch_train, siamese.global_step])
            _, loss_ = sess.run([siamese.optimizer, siamese.loss], feed_dict={siamese.left: image_train1,
                                                                              siamese.right: image_train2,
                                                                              siamese.label: label_train,
                                                                              siamese.training: True})
            print('[train %d, epoch %d, step %d/%d]: loss %.4f'%(train_time, epoch, step,
                                                                 data_sum//siamese.batch_size, loss_),
                  'time %.3fs'%(time.time() - time1))

            if step_ % 10 == 0:
                image_train1, image_train2, label_train = sess.run(
                    [image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_train1,
                                                                                             siamese.right: image_train2,
                                                                                             siamese.label: label_train,
                                                                                             siamese.training: True})
                writer_train.add_summary(summary, step_)
                image_valid1, image_valid2, label_valid = sess.run(
                    [image_batch_valid1, image_batch_valid2, label_batch_valid])
                acc_valid, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_valid1,
                                                                                             siamese.right: image_valid2,
                                                                                             siamese.label: label_valid,
                                                                                             siamese.training: True})
                writer_valid.add_summary(summary, step_)
                print('[epoch %d, step %d/%d]: train acc %.3f, valid acc %.3f'%(step//(data_sum//siamese.batch_size),
                                                                                step%(data_sum//siamese.batch_size),
                                                                                data_sum//siamese.batch_size, acc_train,
                                                                                acc_valid),
                      'time %.3fs'%(time.time()-time1))
            if step_ % 500 == 0:
                print("Save the model Successfully")
                saver.save(sess, "../model/model.ckpt", global_step=step_)
        else:
            step_start = 0
    else:
        print("Save the model Successfully")
        saver.save(sess, "../model/model.ckpt", global_step=step_)
        if not os.path.exists("../result/log"):
            os.makedirs("../result/log")
        f = open("../result/log/train%d.log" % train_time, "w+")
        f.close()

    coord.request_stop()
    coord.join(threads)
