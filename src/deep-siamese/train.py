from model import Siamese
import tensorflow as tf
import config
import time
import heapq
import sys


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


def load_training_set():
    # Load training set.
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode(config.tfrecord_train_path)
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=10240, min_after_dequeue=5120
        )
    return image_batch_train1, image_batch_train2, label_batch_train


def load_valid_set():
    # Load Testing set.
    with tf.name_scope('input_valid'):
        image_valid1, image_valid2, label_valid = read_and_decode(config.tfrecord_valid_path)
        image_batch_valid1, image_batch_valid2, label_batch_valid = tf.train.shuffle_batch(
            [image_valid1, image_valid2, label_valid], batch_size=512, capacity=2560, min_after_dequeue=1280
        )
    return image_batch_valid1, image_batch_valid2, label_batch_valid


def train():
    # network
    batch_size = 512
    siamese = Siamese()

    image_batch_train1, image_batch_train2, label_batch_train = load_training_set()
    image_batch_valid1, image_batch_valid2, label_batch_valid = load_valid_set()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # general setting
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # Recording training process.
        writer_train = tf.summary.FileWriter(config.log_train_path, sess.graph)
        writer_valid = tf.summary.FileWriter(config.log_valid_path, sess.graph)

        last_file = tf.train.latest_checkpoint(config.model_save_path)
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
        # train
        while 1:
            time1 = time.time()
            image_train1, image_train2, label_train, step = sess.run(
                [image_batch_train1, image_batch_train2, label_batch_train, siamese.global_step])
            _, loss_ = sess.run([siamese.optimizer, siamese.loss], feed_dict={siamese.left: image_train1,
                                                                              siamese.right: image_train2,
                                                                              siamese.label: label_train,
                                                                              siamese.training: True})
            print('[epoch %d, step %d/%d]: loss %.6f' % (
            step // (6E5 // batch_size), step % (6E5 // batch_size), 6E5 // batch_size, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step % 10 == 0:
                image_train1, image_train2, label_train = sess.run(
                    [image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_train1,
                                                                                             siamese.right: image_train2,
                                                                                             siamese.label: label_train,
                                                                                             siamese.training: True})
                writer_train.add_summary(summary, step)
                image_valid1, image_valid2, label_valid = sess.run(
                    [image_batch_valid1, image_batch_valid2, label_batch_valid])
                acc_valid, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_valid1,
                                                                                             siamese.right: image_valid2,
                                                                                             siamese.label: label_valid,
                                                                                             siamese.training: True})
                writer_valid.add_summary(summary, step)
                print('[epoch %d, step %d/%d]: train acc %.3f, valid acc %.3f' % (step // (6E5 // batch_size),
                                                                                  step % (6E5 // batch_size),
                                                                                  6E5 // batch_size, acc_train,
                                                                                  acc_valid),
                      'time %.3fs' % (time.time() - time1))
            if step % 500 == 0:
                print("Save the model Successfully")
                saver.save(sess, config.model_name, global_step=step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    train()
