import numpy as np
from sklearn.metrics import confusion_matrix


def quadratic_kappa_score(true_vals, predicted_vals, nb_classes=3):
    """ Encode solubility class
    Parameters
    ----------
    true_vals : numpy array
        true values

    predicted_vals : numpy array

    nb_classes: int
        number of classes

    Return
    ------
    score ([0..1])
    """

    # TODO
    # from here
    # https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook
    # but later it's better to move to something like that
    # https://www.kaggle.com/code/tsmith26/lightgbm-custom-objective-for-weighted-kappa

    conf_m = confusion_matrix(true_vals, predicted_vals)
    w = np.zeros((nb_classes, nb_classes))
    for i_ in range(len(w)):
        for j_ in range(len(w)):
            w[i_][j_] = float(((i_ - j_) ** 2) / (nb_classes - 1) ** 2)

    act_hist = np.zeros([nb_classes])
    for item in true_vals:
        act_hist[item] += 1

    pred_hist = np.zeros([nb_classes])
    for item in predicted_vals:
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
