# A tensorflow implementation for siamese CCR

## 1 Deployment Environment
First, you should install the following libraries:
```
pip install tensorflow-gpu==1.8
pip install opencv-python
pip install pandas
```
Then, decompress the training data and testing data in `data/`, there are 3 `.7z` files, please decompress these three files in the `data/` folder.

## 2 Train and test
`main.py` will generate training set and testing set, train the models, and output testing results. All the above operation will do automatically. If you're happy with the result at one point, please use `ctrl+c` to stop this program.
```
python code/main.py
```

## 3 Reset
`clean.py` will delete all files generated during training and testing.
```
python code/clean.py
```
