import pandas as pd
import config

table = pd.read_csv(config.train_labels_path, encoding="gb2312", header=None)
print(table)
labels = pd.read_csv("gb2312_level1.csv")
print(labels)
chinese = [item[2] for item in labels.values]
id = [item[4] for item in labels.values]
chinese2id = dict(zip(chinese, id))
id2chinese = dict(zip(id, chinese))
results = [chinese2id[item[1]] for item in table.values]
print(results)