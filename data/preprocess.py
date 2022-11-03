import pandas as pd
import sys
import os
from os.path import join

# input files
feature_files = ['cdk_descr.csv', 'substr_count.csv']
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

train_raw_data.sort_index(inplace=True)
test_raw_data.sort_index(inplace=True)
train_indx = list(train_raw_data.index)
test_indx  = list(test_raw_data.index)
train_data = df.loc[train_indx,:]
test_data = df.loc[test_indx,:]
#Sort
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
train_data.to_pickle(os.path.join(folder_put, "X.pk"))
train_raw_data.to_pickle(os.path.join(folder_put, "Y.pk"))
test_data.to_pickle(os.path.join(folder_put, "test.pk"))

sys.stdout.write(f"Data was saved in {folder_put} \n")