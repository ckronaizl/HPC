# initial article on dropout networks: https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
# current code from: https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/14-%20Solving%20overfitting%20in%20neural%20networks/code/solving_overfitting.py
# video on code at: https://www.youtube.com/watch?v=Gf5DO6br0ts&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=14
# This seemed to be more simple than the class implementation; easier to see what network contains
import tensorflow as tf
from tensorflow import keras
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def DropoutNetwork(input_data):
    model = keras.Sequential([
        # input layer; flatten will make a 2D array [[1, 2], [3, 4]] into a 1D array [1, 2, 3, 4]
        #keras.layers.Flatten(input_shape=(input_data.shape[1], input_data.shape[2])),

        # first dense layer
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dropout(0.3),

        # second dense layer
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.3),

        # third dense layer
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model/graph
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
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

    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/epoch_results.png")


if __name__ == '__main__':
    epoch_value = 1

    # load data set
    #(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    nba_train = pd.read_csv("C:/local/nba_stats/NBA_train_83-15.csv", header=None,
                            names=["W/L", "MIN", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
                                   "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "+/-", "POSS", "TS%",
                                   "OER"])

    nba_train_features = nba_train.copy()
    nba_train_labels = nba_train_features.pop("W/L")

    nba_train_features = np.asarray(nba_train_features).astype(np.float)
    nba_train_labels = np.asarray(nba_train_labels).astype(np.float)

    nba_train = pd.read_csv("C:/local/nba_stats/NBA_test_15-20.csv", header=None,
                            names=["W/L", "MIN", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
                                   "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "+/-", "POSS", "TS%",
                                   "OER"])

    nba_test_features = nba_train.copy()
    nba_test_labels = nba_test_features.pop("W/L")

    nba_test_features = np.asarray(nba_test_features).astype(np.double)
    nba_test_labels = np.asarray(nba_test_labels).astype(np.double)

    # convert data set to be between 0 - 1
    #train_images = train_images / 255.0
    #test_images = test_images / 255.0

    # set aside validation images
    #validation_images = train_images[:5000]
    #validation_labels = train_labels[:5000]

    nba_val_features = nba_train_features[31924:]
    nba_val_labels = nba_train_labels[31924:]

    nba_train_features = nba_train_features[:31924]
    nba_train_labels = nba_train_labels[:31924]

    # create a dropout network
    model = DropoutNetwork(nba_train_features)

    # fit our model to data
    start = time.time()
    history = model.fit(train_images, train_labels, epochs=100, validation_data=(validation_images, validation_labels), batch_size=32)
    end = time.time()

    # evaluate model with testing data
    loss, accuracy = model.evaluate(test_images, test_labels)

    with open("./results/No Dropout Results.txt", 'w') as output_stream:
        output_stream.write(f"Loss: {loss}\n")
        output_stream.write(f"Accuracy: {accuracy}\n")

    # view a history of data
    plot_history(history)
    total_time = end - start
    print(f"Time: {total_time}")

