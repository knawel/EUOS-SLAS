import pandas as pd
import sys
import os

# TODO:
# - compare features of train and test sets


# input files: [path, delimeter]
X_file = ["../data/derived/train_cdk.des", ' ']
Y_file = ["../data/raw/train.csv", ',']
X_test_file = ["../data/derived/test_cdk.des", ' ']
folder_put = "../data/preprocessed"

# names of specific columns
y_column_name = "sol_category"
# ID of examples is 'Id'

# read
data = pd.read_csv(X_file[0], delimiter=X_file[1])
data_y = pd.read_csv(Y_file[0], delimiter=Y_file[1])
data_test = pd.read_csv(X_test_file[0], delimiter=X_test_file[1])
sys.stdout.write("Data was imported \n")

# rename and sort
data_y = data_y.loc[:, ['Id', y_column_name]]
data_y.rename(columns={y_column_name: "Y"}, inplace=True)
data.rename(columns={'Title': 'Id'}, inplace=True)
data_test.rename(columns={'Title': 'Id'}, inplace=True)
data_y.sort_values(by='Id')
data.sort_values(by='Id')

# check if the same Id for X and Y
set_id_x = set(data['Id'])
set_id_y = set(data_y['Id'])

diff_size = len(set_id_x - set_id_y)
if diff_size > 0:
    sys.exit('Error. Different set of ID for X and Y dataset')

list_id_x = list(set_id_x)
list_id_y = list(set_id_y)
for i, n in enumerate(list_id_x):
    if n != list_id_y[i]:
        sys.exit(f'Error. Id# {i} are different: {n} and {list_id_y[i]}')

# placeholder for memory reduction
df_train = data
df_train_y = data_y
df_test = data_test

sys.stdout.write("Data was prepared \n")

df_train.to_pickle(os.path.join(folder_put, "X.pk"))
df_train_y.to_pickle(os.path.join(folder_put, "Y.pk"))
df_test.to_pickle(os.path.join(folder_put, "test.pk"))

sys.stdout.write(f"Data was saved in {folder_put} \n")
