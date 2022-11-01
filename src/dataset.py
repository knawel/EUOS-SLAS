import numpy as np
import torch as pt
import pandas as pd
from os.path import join

class MolDataset(pt.utils.data.Dataset):
    def __init__(self, final_dataset_file, y_datafile=None, normal=False):
        super(MolDataset).__init__()
        # Read file with descriptors
        X = pd.read_pickle(final_dataset_file)
        self.X_mat = X.loc[:, X.columns != 'Id'].values
        # Read Y values
        if y_datafile is None:
            self.y = None
        else:
            y = pd.read_pickle(y_datafile)
            self.y = y.loc[:, y.columns != 'Id'].values.flatten()

    def __len__(self):
        return len(self.X_mat)

    def __getitem__(self, index):

        prop = self.X_mat[index, :]
        solubility = self.y[index]
        return prop, solubility
