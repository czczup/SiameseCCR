import shutil
import os

shutil.rmtree("../data/tfrecord")
os.mkdir("../data/tfrecord")

shutil.rmtree("../model")
os.mkdir("../model")

shutil.rmtree("../result")
os.mkdir("../result")