import pickle
from os.path import join
import numpy as np
import pandas as pd
import config
if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "test.pk.zip"))
    X_mat = X.values

    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)

    scaler = model[0]
    lgb = model[1]

    data_sc = scaler.transform(X_mat)
    predicted_vals_onehot = lgb.predict(data_sc)
    predicted_vals = [np.argmax(line) for line in predicted_vals_onehot]
    report = pd.DataFrame({'Id': X.index, 'pred': predicted_vals})
    report.to_csv("submission.csv", index=False)
