# SiameseCCR

## Deployment Environment
This repository relies on the following libraries:
```
pip install tensorflow-gpu==1.8
pip install opencv-python
pip install pandas
```
Then, decompress the training data and testing data in `data/`, there are 3 `.7z` files, please decompress these three files in the `data/` folder.

## Train and test
`main.py` will generate training set and testing set, train the model, and output testing results. All the above operation will do automatically. During training, if you're satisfied with the result at one point, please use `ctrl+c` to stop this program. 
```
python code/main.py
```

## Reset
`clean.py` will delete all files generated during training and testing.
```
python code/clean.py
```

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{chen2020siameseccr,
  title={SiameseCCR: a novel method for one-shot and few-shot Chinese CAPTCHA recognition using deep Siamese network},
  author={Chen, Zhe and Ma, Weifeng and Xu, Nanfan and Ji, Caoting and Zhang, Yulai},
  journal={IET Image Processing},
  volume={14},
  number={12},
  pages={2855--2859},
  year={2020},
  publisher={Wiley Online Library}
}
```
