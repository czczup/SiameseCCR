import time
import os
deviceID = input("please input the device ID: ")
os.environ["CUDA_VISIBLE_DEVICES"] = deviceID

timestamp = str(int(time.time()))[-5:]
ch = input("old or new ?(old/new):")
if ch == "old":
    timestamp = input("please input the id of model:")

train_image_path = "../../res/dataset/train"
train_labels_path = "../../res/dataset/train/labels.txt"
tfrecord_path = "../../res/tfrecord"
tfrecord_train_path = "../../res/tfrecord/classifier_train.tfrecord"
tfrecord_valid_path = "../../res/tfrecord/classifier_valid.tfrecord"
model_save_path = "../../res/model/"+timestamp
model_name = "../../res/model/"+timestamp+"/siamese.ckpt"
log_train_path = "../../res/log/"+timestamp+"/train"
log_valid_path = "../../res/log/"+timestamp+"/valid"
test_seen_image_path = "../../res/dataset/test/seen"
test_seen_labels_path = "../../res/dataset/test/seen/labels.txt"
