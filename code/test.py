import cv2
import sys
import heapq
import time
import os
import pandas as pd


class Character:
    def __init__(self, image, label, feature=None):
        self.image = image
        self.label = label
        self.feature = feature


def load_paths(dataset):
    if dataset == "train" or dataset == "test-seen":
        paths = ["../data/template/seen"]
    elif dataset == "test-unseen":
        paths = ["../data/template/unseen"]
    else:
        paths = ["../data/template/seen", "../data/template/unseen"]
    return paths


def load_labels(dataset):
    if dataset == "train":
        table = pd.read_csv("../data/train/labels.txt", header=None, encoding='gb2312')
        labels = [item[1] for item in table.values]
    elif dataset == "test-seen":
        table = pd.read_csv("../data/test/seen/labels.txt", header=None, encoding='gb2312')
        labels = [item[1] for item in table.values]
    elif dataset == "test-unseen":
        table = pd.read_csv("../data/test/unseen/labels.txt", header=None, encoding='gb2312')
        labels = [item[1] for item in table.values]
    else:
        table1 = pd.read_csv("../data/test/seen/labels.txt", header=None, encoding='gb2312')
        table2 = pd.read_csv("../data/test/unseen/labels.txt", header=None, encoding='gb2312')
        labels = [item[1] for item in table1.values]+[item[1] for item in table2.values]
    return labels


def load_test_data(dataset, labels):
    if dataset == "train":
        paths = ["../data/train"]
        amount = 15020
    elif dataset == "test-seen":
        paths = ["../data/test/seen"]
        amount = 10000
    elif dataset == "test-unseen":
        paths = ["../data/test/unseen"]
        amount = 10000
    else:
        paths = ["../data/test/seen", "../data/test/unseen"]
        amount = 10000

    character_list = []
    cnt = 0
    for path in paths:
        for i in range(amount):
            name = str(i).zfill(4) if not dataset == "train" else str(i).zfill(5)
            image = cv2.imread(path+"/"+name+".jpg", cv2.IMREAD_GRAYSCALE)
            image = image[2:46, 2:46]
            image = image.reshape([44, 44, 1])/255.0
            character = Character(image, labels[cnt])
            character_list.append(character)
            cnt += 1
    return character_list


def load_template_list(paths, sess, siamese):
    template_list = []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith("jpg"):
                image = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE)
                image = image[2:46, 2:46]
                image = image.reshape([44, 44, 1])/255.0
                label = file[0]
                feature = sess.run(siamese.right_output, feed_dict={siamese.right: [image],
                                                                    siamese.training: False})[0]
                template = Character(image, label, feature)
                template_list.append(template)
    print("字符模板加载完成")
    print("字符模板总数为%d" % len(template_list))
    return template_list


def predict(character_list, template_list, sess, siamese):
    time1 = time.time()
    pred_characters = []
    true_characters = []
    length = len(character_list)
    for index, character in enumerate(character_list):
        feature = sess.run(siamese.left_output, feed_dict={siamese.left: [character.image], siamese.training: False})
        prediction = sess.run(siamese.test_y_hat, feed_dict={siamese.image_feature: feature,
                                             siamese.template_feature: [template.feature for template in template_list],
                                             siamese.training: False})
        min_list = heapq.nsmallest(10, range(len(prediction)), prediction.take)
        pred_character = [template_list[item].label for item in min_list]
        pred_characters.append(pred_character)
        true_characters.append(character.label)

        sys.stdout.write('\r>> Testing image %d/%d' % (index+1, length))
        sys.stdout.flush()
    time2 = time.time()
    print("\nUsing time:", "%.2f"%(time2-time1)+"s")
    return pred_characters, true_characters


def calculate_accuracy(dataset, true_characters, pred_characters, train_time):
    count_top1 = 0
    count_top5 = 0
    count_top10 = 0
    if dataset == "train":
        f = open("../result/result-train%d.txt"%train_time, "w+")
    elif dataset == "test-seen":
        f = open("../result/result-test-seen%d.txt"%train_time, "w+")
    elif dataset == "test-unseen":
        f = open("../result/result-test-unseen%d.txt"%train_time, "w+")
    elif dataset == "test":
        f = open("../result/result-test%d.txt"%train_time, "w+")

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
    f.write(str(count_top1/len(true_characters))+",")
    f.write(str(count_top5/len(true_characters))+",")
    f.write(str(count_top10/len(true_characters)))
    print("top1 acc:", count_top1/len(true_characters))
    print("top5 acc:", count_top5/len(true_characters))
    print("top10 acc:", count_top10/len(true_characters))
    f.close()


def test(siamese, sess, dataset, train_time):
    # 根据测试的数据集加载对于的匹配模板
    paths = load_paths(dataset)
    template_list = load_template_list(paths, sess, siamese)
    labels = load_labels(dataset)
    character_list = load_test_data(dataset, labels)
    pred_characters, true_characters = predict(character_list, template_list, sess, siamese)
    calculate_accuracy(dataset, true_characters, pred_characters, train_time)
