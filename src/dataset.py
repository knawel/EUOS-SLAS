import numpy as np
import torch as pt
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler, Normalizer


def encode_y(target, nb_classes=3):
    """ Encode solubility class
    Parameters
    ----------
    y : int
        solubility class
    Return
    ------
    1/0 vector (numpy)
    """
    res = np.array(np.eye(nb_classes)[target])#, dtype=int)
    return res



class MolDataset(pt.utils.data.Dataset):
    def __init__(self, final_dataset_file, y_datafile=None, normal=False):
        super(MolDataset).__init__()
        # Read file with descriptors
        X = pd.read_pickle(final_dataset_file)
        self.X_mat = np.array(X.values, dtype=np.float32)
        # Read Y values
        if y_datafile is None:
            self.y = None
        else:
            y = pd.read_pickle(y_datafile)
            self.y = np.array(y.values.flatten(), dtype=int)
        
        if normal:
            self.scaler = Normalizer()
            self.X_mat = self.scaler.fit_transform(self.X_mat)

    def __len__(self):
        return len(self.X_mat)
    
    def get_scaler(self):
        return self.scaler

    def __getitem__(self, index):
        prop = self.X_mat[index, :]
        solubility = self.y[index]
        return prop, solubility, encode_y(solubility, 3)
