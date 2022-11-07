import pickle
from os.path import join

import pandas as pd
import config
if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "test.pk.zip"))
    X_mat = X.values

    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)
    predicted_vals = model.predict(X_mat)
    report = pd.DataFrame({'Id': X.index, 'pred': predicted_vals})
    report.to_csv("submission.csv", index=False)
