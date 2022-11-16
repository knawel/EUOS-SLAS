import pandas as pd
import sys
import os
from os.path import join
import numpy as np

# input files
# substr_count.csv
# compounds_desalt_3dH.oddescriptors.csv
# compounds_desalt_3dH.csv
feature_files = ['compounds_desalt_3dH.csv', 'substr_count.csv']

train_raw_file = "../data/raw/train.csv"
test_raw_file = "../data/raw/test.csv"
folder_put = "../data/preprocessed/"

# names of specific columns
y_column_name = "sol_category"

# read raw data
train_raw_data = pd.read_csv(train_raw_file, index_col='Id')
test_raw_data = pd.read_csv(test_raw_file, index_col='Id')
# rename
train_raw_data = train_raw_data.loc[:, [y_column_name]]
train_raw_data.rename(columns={y_column_name: "Y"}, inplace=True)

# Read features
features = []
for f in feature_files:

    dfile = join("../data/derived/", f)
    data = pd.read_csv(dfile)
    data.dropna(axis=1, inplace=True)

    if not ('Id' in data.columns):
        print('error')
    else:
        print('all good')
    features.append(data)

# combine features
df = features[0]
for d in features[1:]:
    df = pd.merge(df, d, on='Id')

df.set_index('Id', inplace=True)
if df.isnull().values.any():
    sys.stdout.write("Merge failed\n")
else:
    sys.stdout.write("Data was imported \n")

# remove features with all same values
sys.stdout.write(f"The number of features: {df.shape[1]}\n")
cols = df.select_dtypes([np.number]).columns
std = df[cols].std()
cols_to_drop = std[std < 0.001].index
df.drop(cols_to_drop, axis=1, inplace=True)
sys.stdout.write(f"The number of features after removing constants: {df.shape[1]}\n")

# split
train_raw_data.sort_index(inplace=True)
test_raw_data.sort_index(inplace=True)
train_indx = list(train_raw_data.index)
test_indx = list(test_raw_data.index)
train_data = df.loc[train_indx, :]
test_data = df.loc[test_indx, :]

# sort
train_data.sort_index(inplace=True)
test_data.sort_index(inplace=True)

# check if the same Id for X and Y
list_id_x = list(train_data.index)
list_id_y = list(test_data.index)
if (list_id_x == train_indx) and (list_id_y == test_indx):
    sys.stdout.write("Data was splitted\n")
else:
    sys.stdout.write("errors with indexing\n")


# save files
train_data.to_pickle(os.path.join(folder_put, "X.pk.zip"))
train_raw_data.to_pickle(os.path.join(folder_put, "Y.pk.zip"))
test_data.to_pickle(os.path.join(folder_put, "test.pk.zip"))

sys.stdout.write(f"Data was saved in {folder_put} \n")