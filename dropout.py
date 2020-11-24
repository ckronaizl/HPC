# initial article on dropout networks: https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
# current code from: https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/14-%20Solving%20overfitting%20in%20neural%20networks/code/solving_overfitting.py
# video on code at: https://www.youtube.com/watch?v=Gf5DO6br0ts&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=14
# This seemed to be more simple than the class implementation; easier to see what network contains
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import time
import random
from pprint import pprint


def DropoutNetwork():
    dropout_rate = 0.5
    model = keras.Sequential([

        keras.layers.Dense(80, activation="relu"),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(80, activation="relu"),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(120, activation="relu"),
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


def plot_history(history):

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

    plt.savefig("Dropout Results.png")
    plt.close()


if __name__ == '__main__':

    nba_train = pd.read_csv("test_train_data/NBA_train_83-15.csv", header=None, na_values="-",
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

    slice_value = 20000
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
    model = DropoutNetwork()

    # fit our model to data
    train_start = time.time()
    history = model.fit(nba_train_features, nba_train_labels, epochs=3, validation_data=(nba_val_features, nba_val_labels), batch_size=64)
    train_end = time.time()

    # evaluate model with testing data
    predict_start = time.time()
    predictions = model(nba_test_features, training=False)
    predict_end = time.time()

    # view a history of data
    plot_history(history)
    train_time = train_end - train_start
    test_time = predict_end - predict_end
    print(f"Training time: {train_time}")
    print(f"Testing time: {test_time}")

    # save model as png
    keras.utils.plot_model(model, to_file="model_visual.png")
    keras.utils.plot_model(model, to_file="model_visual_shapes.png", show_shapes=True)
    keras.utils.plot_model(model, to_file="model_visual_layers.png", show_layer_names=True)
    keras.utils.plot_model(model, to_file="model_visual_shapes_layers.png", show_shapes=True, show_layer_names=True)



