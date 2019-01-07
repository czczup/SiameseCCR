import tensorflow as tf
import os
import random
import sys
import cv2
import config
# The number of images in the validation set.
_NUM_VALID = 20000

def _get_filenames_and_classes(dataset_dir):
    """ Get all classes and filenames. """
    import pandas as pd
    import os
    import random

    table = pd.read_csv("gb2312_level1.csv")
    character_dict = {}

    for character in table.values:
        character_dict[character[2]] = [0, [], character[4]]

    label = pd.read_csv("../../res/dataset/train/labels.txt", header=None, encoding="gb2312")
    ids = [item[0] for item in label.values]
    characters = [item[1] for item in label.values]
    id2character = dict(zip(ids, characters))

    file_list = []
    for file in os.listdir("../../res/dataset/train"):
        if not file.endswith("txt"):
            file_list.append(file)

    for file in file_list:
        filename = int(file.split('.')[0])
        character = id2character[filename]
        character_dict[character][0] += 1
        character_dict[character][1].append(file)

    image_filenames = []
    class_names = []
    for key, value in character_dict.items():
        filanames = value[1]
        classname = value[2]
        for filename in filanames:
            image_filenames.append(filename)
            class_names.append(classname)
    return image_filenames,class_names

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image':bytes_feature(image_data),
        'label':int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names,labels_filename):
    """ Write value-label into labels.txt """
    with tf.gfile.Open(labels_filename,"w") as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label,class_name))

def _convert_dataset(split_name,filenames,class_names_to_ids,dataset_dir):
    """ Convert data to TFRecord format. """
    assert split_name in ['train','valid']
    with tf.Graph().as_default():
        with tf.Session() as sess:
            output_filename = os.path.join(dataset_dir,"classifier_%s.tfrecord" % split_name)
            tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
            for i in range(len(filenames)):
                image = cv2.imread("../../res/dataset/train/"+filenames[i], cv2.IMREAD_GRAYSCALE)
                # image = image[:,0:48]
                image = image.tobytes()
                class_id = filename_to_ids[filenames[i]]
                example = image_to_tfexample(image, class_id)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Converting image %d/%d'%(i+1, len(filenames)))
                sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # Get all images and classes.
    image_filenames, class_names = _get_filenames_and_classes(config.image_pair_path)
    # Change classes into Python dict
    class_names_to_ids = dict(zip(class_names,range(len(class_names))))
    filename_to_ids = dict(zip(image_filenames, class_names))
    # Shuffle the image set.
    random.seed(0)
    random.shuffle(image_filenames)
    # Divide the data into training sets and validation sets.
    training_filenames = image_filenames[_NUM_VALID:]
    validation_filenames = image_filenames[:_NUM_VALID]
    # Convert data to TFRecord format.
    _convert_dataset('train', training_filenames, class_names_to_ids, config.tfrecord_path)
    _convert_dataset('valid', validation_filenames, class_names_to_ids, config.tfrecord_path)
