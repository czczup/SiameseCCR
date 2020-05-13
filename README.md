# A TensorFlow implementation of SiameseCCR

## 1 Deployment Environment
This repository relies on the following libraries:
```
pip install tensorflow-gpu==1.8
pip install opencv-python
pip install pandas
```
Then, decompress the training data and testing data in `data/`, there are 3 `.7z` files, please decompress these three files in the `data/` folder.

## 2 Train and test
`main.py` will generate training set and testing set, train the model, and output testing results. All the above operation will do automatically. During training, if you're satisfied with the result at one point, please use `ctrl+c` to stop this program. 
```
python code/main.py
```

## 3 Reset
`clean.py` will delete all files generated during training and testing.
```
python code/clean.py
```
