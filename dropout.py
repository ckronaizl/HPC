# initial article on dropout networks: https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
# current code from: https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/14-%20Solving%20overfitting%20in%20neural%20networks/code/solving_overfitting.py
# video on code at: https://www.youtube.com/watch?v=Gf5DO6br0ts&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=14
# This seemed to be more simple than the class implementation; easier to see what network contains
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random

import time
import shutil
import datetime
import sys
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_time():
    time = datetime.datetime.now()
    year = time.strftime("%Y")  # full year
    month = time.strftime("%b")  # short-hand abbreviation
    day = time.strftime("%d")  # 0-padded value
    hour = time.strftime("%I")  # 12 hour format with padded 0
    minute = time.strftime("%M")  # minute with padded 0
    second = time.strftime("%S")
    am_pm = time.strftime("%p").lower()  # am or pm, lowercase

    date = f"{month}-{day}-{year} {hour}:{minute}:{second}{am_pm}"
    return date


def DropoutNetwork(one, two, three):
    dropout_rate = 0.5
    model = keras.Sequential([

        keras.layers.Dense(one, activation="relu"),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(two, activation="relu"),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(three, activation="relu"),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile model/graph
    # Stochastic Gradient Descent (SGD) article: https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
    # why momentum is important: https://medium.com/analytics-vidhya/why-use-the-momentum-optimizer-with-minimal-code-example-8f5d93c33a53
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    return model


def plot_history(history, one, two, three):

    # create a new figure with two subplots
    fig, axs = plt.subplots(2)

    # history is a dictionary
    # history.history has four values: accuracy, val_accuracy, loss, val_loss
    # create new plot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    return plt


def save_plot_pickle_results(plot, predictions, history, one, two, three):
    save_path = os.getcwd()
    save_path = os.path.join(save_path, "results", str(one), str(two), str(three))
    os.makedirs(save_path, exist_ok=True)

    plot_path = os.path.join(save_path, f"{one}.{two}.{three}.graph.png")
    pickle_path = os.path.join(save_path, f"{one}.{two}.{three}.predictions.pkl")
    history_path = os.path.join(save_path, f"{one}.{two}.{three}.history.pkl")

    plot.savefig(plot_path)

    # save object to file
    pickle.dump(predictions, open(pickle_path, 'wb'))
    pickle.dump(history.history, open(history_path, 'wb'))


if __name__ == '__main__':

    nba_train = pd.read_csv("test_train_data/NBA_train_83-15.csv", header=None,
                            names=["W/L", "MIN", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
                                   "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "+/-", "POSS", "TS%",
                                   "OER"])

    nba_train_features = nba_train.copy()
    nba_train_labels = nba_train_features.pop("W/L")

    nba_train_features = np.asarray(nba_train_features).astype(np.float)
    nba_train_labels = np.asarray(nba_train_labels).astype(np.float)

    nba_test = pd.read_csv("test_train_data/NBA_test_15-20.csv", header=None,
                           names=["W/L", "MIN", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
                                  "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "+/-", "POSS", "TS%",
                                  "OER"])

    nba_test_features = nba_test.copy()
    nba_test_labels = nba_test_features.pop("W/L")

    nba_test_features = np.asarray(nba_test_features).astype(np.double)
    nba_test_labels = np.asarray(nba_test_labels).astype(np.double)

    # use 20% as validation
    slice_value = int(len(nba_train_labels) * 0.2)
    nba_val_features = nba_train_features[:slice_value]
    nba_val_labels = nba_train_labels[:slice_value]

    nba_train_features = nba_train_features[slice_value:]
    nba_train_labels = nba_train_labels[slice_value:]

    # this website talks about needing to shuffle values, otherwise network fixates on first values seen
    # https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
    # shuffle numpy arrays: https://stackoverflow.com/a/49755088
    indices = np.arange(nba_train_features.shape[0])
    np.random.shuffle(indices)
    nba_train_features = nba_train_features[indices]
    nba_train_labels = nba_train_labels[indices]

    # create a dropout network
    # start, end, step
    """
    I have no idea if attempting to determine the best number of nodes like this is the
        best implementation, but I guess that's what HPC is for  ¯\_(ツ)_/¯ 
    """
    for input_layer in range(10, 151, 10):
        for hidden_layer_one in range(10, 151, 10):
            for hidden_layer_two in range(10, 151, 10):
                print(f"Layer 1 = {input_layer}")
                print(f"Layer 2 = {hidden_layer_one}")
                print(f"Layer 3 = {hidden_layer_two}")
                print("\n")
                model = DropoutNetwork(input_layer, hidden_layer_one, hidden_layer_two)

                # fit our model to data
                history = model.fit(nba_train_features, nba_train_labels, epochs=100, validation_data=(nba_val_features, nba_val_labels), batch_size=64, verbose=0)

                # evaluate model with testing data
                # loss, accuracy = model.predict(nba_test_features, nba_test_labels)
                predictions = model.predict(nba_test_features, workers=4, use_multiprocessing=True)

                # view a history of data
                plot = plot_history(history, input_layer, hidden_layer_one, hidden_layer_two)

                save_plot_pickle_results(plot, predictions, history, input_layer, hidden_layer_one, hidden_layer_two)
                exit(0)
