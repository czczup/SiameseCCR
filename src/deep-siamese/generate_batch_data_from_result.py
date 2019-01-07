import random
import cv2
import numpy as np
from multiprocessing import Process
import time
import os
import config
import pandas as pd
import shutil

class Character:
    def __init__(self):
        self.filenames = []
        self.stroke_number = None
        self.id = None


table = pd.read_csv("gb2312_level1.csv")
print(table)
id = [item[4] for item in table.values]
chinese = [item[2] for item in table.values]
stroke = [item[3] for item in table.values]
chinese2stroke = dict(zip(chinese,stroke))
chinese2id = dict(zip(chinese, id))
assert len(chinese2stroke) == 3755, "error length"

label = pd.read_csv("../../res/dataset/train/labels.txt", header=None, encoding="gb2312")
print(label)
characters = {}
for index, ch in label.values:
    id = chinese2id[ch]
    filename = str(index).zfill(5)+".jpg"
    if id in characters:
        characters[id].filenames.append(filename)
    else:
        characters[id] = Character()
        characters[id].filenames.append(filename)
        characters[id].stroke_number = chinese2stroke[ch]
        characters[id].id = id

for key, value in characters.items():
    assert len(value.filenames) == 4, "error"

results = pd.read_csv("results.txt", header=None)
top10error = [item for item in results.values]
assert len(top10error)==15020, "error length of results"

def get_positive_and_negative_batch(number):
    time1 = time.time()
    if not os.path.exists(config.image_pair_path+"/0/"):
        os.makedirs(config.image_pair_path+"/0/")
    if not os.path.exists(config.image_pair_path+"/1/"):
        os.makedirs(config.image_pair_path+"/1/")
    for i in range(50000):
        image1, image2, stroke = get_positive_pair()
        image_1 = np.concatenate([image1, image2], axis=1)
        save_path = config.image_pair_path+"/1/"+str(i)+"_"+str(number)+"_"+stroke+".jpg"
        cv2.imwrite(save_path, image_1)
        image1, image2, stroke = get_negative_pair()
        image_2 = np.concatenate([image1, image2], axis=1)
        save_path = config.image_pair_path+"/0/"+str(i)+"_"+str(number)+"_"+stroke+".jpg"
        cv2.imwrite(save_path, image_2)
        print("Generating character pairs:"+str(i+1)+"/"+str(50000))
    else:
        time2 = time.time()
        print(str(time2-time1)+"s")


def get_positive_pair():  # 获取正样本对
    index = np.random.randint(0, 3755)  # 随机产生汉字的编号
    image1_index = np.random.randint(0, 4)
    image2_index = np.random.randint(0, 4)
    path1 = config.train_image_path+"/"+characters[index].filenames[image1_index]
    path2 = config.train_image_path+"/"+characters[index].filenames[image2_index]
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    stroke = str(characters[index].stroke_number)+"_"+str(characters[index].stroke_number)
    return image1, image2, stroke


def get_negative_pair():  # 获取负样本对
    index = np.random.randint(0, 15020) # 随机产生汉字的编号
    image_index = np.random.randint(0, 5)
    idx1 = chinese2id[top10error[index][0]] # 第一个字的id
    idx2 = chinese2id[top10error[index][1][image_index]] # 第二个字的id
    path1 = config.train_image_path+"/"+characters[idx1].filenames[np.random.randint(0,4)]
    path2 = config.train_image_path+"/"+characters[idx2].filenames[np.random.randint(0,4)]
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    stroke = str(characters[idx1].stroke_number)+"_"+str(characters[idx2].stroke_number)
    return image1, image2, stroke


if __name__=='__main__':
    shutil.rmtree('../../res/image_pair/1')
    print("1 delete successful")
    shutil.rmtree('../../res/image_pair/0')
    print("0 delete successful")
    p1 = Process(target=get_positive_and_negative_batch, args=(0,))
    p2 = Process(target=get_positive_and_negative_batch, args=(1,))
    p3 = Process(target=get_positive_and_negative_batch, args=(2,))
    p4 = Process(target=get_positive_and_negative_batch, args=(3,))
    p5 = Process(target=get_positive_and_negative_batch, args=(4,))
    p6 = Process(target=get_positive_and_negative_batch, args=(5,))
    pool = [p1, p2, p3, p4, p5, p6]
    for p in pool:
        p.start()
