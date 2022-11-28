import pickle
import sys

import pandas as pd
from os.path import join
import numpy as np

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import config


def calc_custom_kappa(actuals, preds, n=3):
    # from here
    # https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook
    # but later it's better to move to something like that
    # https://www.kaggle.com/code/tsmith26/lightgbm-custom-objective-for-weighted-kappa
    conf_m = confusion_matrix(actuals, preds)
    w = np.zeros((n, n))
    for i_ in range(len(w)):
        for j_ in range(len(w)):
            w[i_][j_] = float(((i_ - j_) ** 2) / (n - 1) ** 2)

    act_hist = np.zeros([n])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([n])
    for item in preds:
        pred_hist[item] += 1
    e_mat = np.outer(act_hist, pred_hist)

    e_mat = e_mat / e_mat.sum()
    conf_m = conf_m / conf_m.sum()

    num = 0
    den = 0
    for i_ in range(len(w)):
        for j_ in range(len(w)):
            num += w[i_][j_] * conf_m[i_][j_]
            den += w[i_][j_] * e_mat[i_][j_]

    weighted_kappa = (1 - (num / den))
    return weighted_kappa

if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "X.pk.zip"))
    y = pd.read_pickle(join(data_folder, "Y.pk.zip"))

    X_mat = X.values
    y_vec = y.values.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.12, random_state=123)
    sys.stdout.write(f"The number of features: {X_mat.shape[1]}\n")
    sys.stdout.write(f"TRAIN \
    0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}, 2: {np.sum(y_train == 2)}\n")

    sys.stdout.write(f"TEST  \
    0: {np.sum(y_test == 0)}, 1: {np.sum(y_test == 1)}, 2: {np.sum(y_test == 2)}\n")

    # Dataset

    # scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    train_data = lgb.Dataset(X_train_sc, label=y_train)
    validation_data = lgb.Dataset(X_test_sc, label=y_test)

    # setting up the parameters
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'multiclassova',
        'metric': 'multi_logloss',
        'num_class': 3,
        'verbose': -1,
        'is_unbalance': True}

    # params['max_depth']=10

    # scores = []
    # d_step = 70
    # n_s = 25
    # for i in range(n_s):
    #     steps = (i + 1) * d_step
    #     sys.stdout.write(f'iter {i+1}, steps = {steps}\n')
    #     bst = lgb.train(params, train_data, steps)  # , feval=kappa_metric)
    #     y_predicted = bst.predict(X_test_sc)
    #     ypr = [np.argmax(line) for line in y_predicted]
    #     k_score = cohen_kappa_score(y_test, ypr)
    #     kk_score = calc_custom_kappa(y_test, ypr)
    #     f1_scores = f1_score(y_test, ypr, average=None)
    #     s = np.append(f1_scores, k_score)
    #     s = np.append(s, kk_score)
    #     scores.append(s)
    #
    # scores = np.array(scores)
    # print(scores)
    # # with open('model.pickle', 'wb') as handle:
    # #     pickle.dump(best_model[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # plt.plot(np.arange(1, n_s + 1) * d_step, scores[:, 3], '-o', label="Cohen Kappa")
    # plt.plot(np.arange(1, n_s + 1) * d_step, scores[:, 4], '-o', label="Quadratic Kappa")
    # plt.plot(np.arange(1, n_s + 1) * d_step, scores[:, 0], '-.', label='F1 for "0"')
    # plt.plot(np.arange(1, n_s + 1) * d_step, scores[:, 1], '-.', label='F1 for "1"')
    # plt.legend()
    # plt.savefig("scores_vs_steps.png")
    #
    # sys.stdout.write('Figure was saved\n')

    # use the model with optimal nsteps
    n_steps = 150
    sys.stdout.write(f'Making model for n steps = {n_steps}\n')

    bst = lgb.train(params, train_data, n_steps)  # , feval=kappa_metric)
    y_predicted = bst.predict(X_test_sc)
    ypr = [np.argmax(line) for line in y_predicted]
    k_score = cohen_kappa_score(y_test, ypr)
    kk_score = calc_custom_kappa(y_test, ypr)
    f1_scores = f1_score(y_test, ypr, average=None)
    s = np.append(f1_scores, k_score)
    s = np.append(s, kk_score)
    sys.stdout.write('Scores\n')
    print(s)

    with open('model.pickle', 'wb') as handle:
        pickle.dump([scaler, bst], handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write('Model was saved.\n')








