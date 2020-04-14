import pandas as pd

TRAIN_ALL_PATH = "data/train_all.csv"

csv = pd.read_csv(TRAIN_ALL_PATH)

train = csv.loc[csv['sample'] != 10]
test = csv.loc[csv['sample'] == 10]

train.to_csv("data/train_training.csv", index=False)
test.to_csv("data/train_testing.csv", index=False)

print("Saved")