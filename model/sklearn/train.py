import pickle

import pandas as pd
from os.path import join
import os

# Sk learn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import config

if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "X.pk"))
    y = pd.read_pickle(join(data_folder, "Y.pk"))

    # X_mat = X.loc[:, X.columns != 'Id'].values
    props = ['ALogP', 'ALogp2', 'AMR', 'nAcid', 'naAromAtom', 'nAromBond',
             'nBase', 'nB', 'nHBDon', 'nHBAcc', 'MLogP', 'nRotB', 'TopoPSA', 'MW',
             'XLogP']
    X_mat = X.loc[:, props].values

    y_vec = y.loc[:, y.columns != 'Id'].values.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.2, random_state=123)

    # collection of classifiers
    classifiers = config.classifiers

    preprocessor = StandardScaler()
    best_model = [None, 0]

    for cls in classifiers:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifiers[cls])
        ])
        model = pipeline.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = f1_score(predictions, y_test, average=None)
        score2 = cohen_kappa_score(predictions, y_test)
        print(f'Model:{cls}; score:{score2}')

        sum_score = score[0] + score[1]
        if score2 > best_model[1]:
            best_model = [model, score2]
    print(best_model)
    with open('model.pickle', 'wb') as handle:
        pickle.dump(best_model[0], handle, protocol=pickle.HIGHEST_PROTOCOL)





