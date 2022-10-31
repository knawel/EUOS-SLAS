import pickle
from os.path import join

import pandas as pd
import config
if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "test.pk"))
    props = ['ALogP', 'ALogp2', 'AMR', 'nAcid', 'naAromAtom', 'nAromBond',
             'nBase', 'nB', 'nHBDon', 'nHBAcc', 'MLogP', 'nRotB', 'TopoPSA', 'MW',
             'XLogP']
    X_mat = X.loc[:, props].values

    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)
    predicted_vals = model.predict(X_mat)
    report = pd.DataFrame({'Id': X.loc[:, 'Id'].values, 'pred': predicted_vals})
    report.to_csv("submission.csv", index=False)
