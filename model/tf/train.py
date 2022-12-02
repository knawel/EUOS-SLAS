# NN
import tensorflow as tf
from tensorflow import keras
# ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from src.scoring import quadratic_kappa_score
from config import config_runtime

# PLOT
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# GENERAL
import os
from os.path import join
import sys


def average_recall(y_true, y_pred):
    # Get indexes of both labels and predictions
    labels = tf.argmax(y_true, axis=1)
    predictions = tf.argmax(y_pred, axis=1)
    # Get confusion matrix from labels and predictions
    confusion_matrix = tf.math.confusion_matrix(labels, predictions)
    # Get number of all true positives in each class
    all_true_positives = tf.linalg.diag_part(confusion_matrix)
    # Get number of all elements in each class
    all_class_sum = tf.reduce_sum(confusion_matrix, axis=1)
    # Get rid of classes that don't show in batch
    mask = tf.not_equal(all_class_sum, tf.constant(0))
    all_true_positives = tf.boolean_mask(all_true_positives, mask)
    all_class_sum = tf.boolean_mask(all_class_sum, mask)

    # print("confusion_matrix:\n {},\n all_true_positives:\n {},\n all_class_sum:\n {}".format(
    #                                         confusion_matrix, all_true_positives, all_class_sum))
    # Average TruePositives / TotalElements wrt all classes that show in batch
    return tf.reduce_mean(all_true_positives / all_class_sum)



def train(config_runtime):
    # from src.logger import Logger
    print("TensorFlow version:", tf.__version__)
    data_folder = "../../data/preprocessed/"

    # read datasets
    X = pd.read_pickle(join(data_folder, "X.pk.zip"))
    y = pd.read_pickle(join(data_folder, "Y.pk.zip"))
    X_mat = X.values
    y_vec = y.values.flatten()
    scaler = StandardScaler()
    scaler.fit(X_mat)
    X_mat = scaler.transform(X_mat)
    x_train, x_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.12, random_state=123)
    sys.stdout.write(f"The number of features: {X_mat.shape[1]}\n")
    sys.stdout.write(f"TRAIN \
    0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}, 2: {np.sum(y_train == 2)}\n")

    sys.stdout.write(f"TEST  \
    0: {np.sum(y_test == 0)}, 1: {np.sum(y_test == 1)}, 2: {np.sum(y_test == 2)}\n")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=config_runtime['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[average_recall])#'categorical_accuracy'])

    class_weight = {0: 1.2,
                    1: 1.2,
                    2: 0.1}

    y_train_oh = tf.one_hot(y_train, 3).numpy()
    y_test_oh = tf.one_hot(y_test, 3).numpy()
    history = model.fit(x_train, y_train_oh,
                        validation_data=(x_test, y_test_oh),
                        epochs=config_runtime['num_epochs'],
                        class_weight=class_weight)

    # evaluate model
    RE = model.predict(x_test)
    results = tf.math.argmax(RE, axis=1).numpy()
    cm = confusion_matrix(results, y_test)
    score2 = quadratic_kappa_score(results, y_test)
    print(cm)
    print(score2)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm / np.sum(cm, axis=0), annot=True,
                fmt='.2%', cmap='Blues')
    fig.savefig("cm.png")

    fig, axs = plt.subplots(2, figsize=(4, 3), sharex=True)
    fig.suptitle("loss")
    axs[0].plot(history.history['loss'], c='blue')
    axs[1].plot(history.history['val_loss'], c='red')
    fig.savefig("loss.png")

    fig, axs = plt.subplots(2, figsize=(4, 3), sharex=True)
    fig.suptitle("average recall")
    axs[0].plot(history.history['average_recall'], c='blue')
    axs[1].plot(history.history['average_recall'], c='red')
    fig.savefig("recall.png")


    print("Done!")

    # pt.save(model.state_dict(), "model.pt")
    #
    # # save scaler
    # scaler = dataset.get_scaler()
    # scaler_filename = "scaler.save"
    # joblib.dump(scaler, scaler_filename)
    #

if __name__ == '__main__':
    # train model
    train(config_runtime)

