import os
deviceID = input("please input the device ID: ")
os.environ["CUDA_VISIBLE_DEVICES"] = deviceID

# import time
# ch = input("旧模型还是新模型(old/new):")
# if ch == "old":
#     timestamp = input("请输入编号:")
# else:
# timestamp = str(int(time.time()))[-5:]

timestamp = "60851"

image_pair_path = "../../res/image_pair"
train_image_path = "../../res/dataset/train"
train_labels_path = "../../res/dataset/train/labels.txt"
tfrecord_path = "../../res/tfrecord"
tfrecord_train_path = "../../res/tfrecord/siamese_train.tfrecord"
tfrecord_valid_path = "../../res/tfrecord/siamese_valid.tfrecord"
model_save_path = "../../res/model/"+timestamp
model_name = "../../res/model/"+timestamp+"/siamese.ckpt"
log_train_path = "../../res/log/"+timestamp+"/train"
log_valid_path = "../../res/log/"+timestamp+"/valid"
test_seen_image_path = "../../res/dataset/test/seen"
test_seen_labels_path = "../../res/dataset/test/seen/labels.txt"
test_unseen_image_path = "../../res/dataset/test/unseen"
test_unseen_labels_path = "../../res/dataset/test/unseen/labels.txt"
seen_template_path = "../../res/dataset/template/seen"
unseen_template_path = "../../res/dataset/template/unseen"