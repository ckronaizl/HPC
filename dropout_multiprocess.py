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
import multiprocessing as mp

import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    plt.savefig(f"results/{one}/{two}/{three}/{one}.{two}.{three}.graph.png")
    plt.close()

    return plt


def path_exists(one, two, three):
    save_path = os.getcwd()
    save_path = os.path.join(save_path, "results", str(one), str(two), str(three))
    if os.path.exists(save_path):
        return True
    else:
        return False


def pickle_results(plot, predictions, history, one, two, three):
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


def multi_process_network(iteration,
                          nba_train_features, nba_train_labels,
                          nba_val_features, nba_val_labels,
                          nba_test_features):
    # do not execute dropout network if we have done this already path exists

    input_layer = iteration[0]
    hidden_layer_one = iteration[1]
    hidden_layer_two = iteration[2]
    model = DropoutNetwork(input_layer, hidden_layer_one, hidden_layer_two)

    # fit our model to data
    history = model.fit(nba_train_features, nba_train_labels, epochs=100, validation_data=(nba_val_features, nba_val_labels), batch_size=64, verbose=0)

    # evaluate model with testing data
    # loss, accuracy = model.predict(nba_test_features, nba_test_labels)
    predictions = model.predict(nba_test_features, workers=4, use_multiprocessing=True)

    # view a history of data
    plot = plot_history(history, input_layer, hidden_layer_one, hidden_layer_two)

    pickle_results(plot, predictions, history, input_layer, hidden_layer_one, hidden_layer_two)
    with open("results/completed_iterations.txt", 'a') as output_stream:
        output_stream.write(f"Completed {iteration}\n")


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

    """
    layers[0], layers[1], layers[2],
    nba_train_features, nba_train_labels, nba_val_features,
    nba_val_labels, nba_test_features
    """

    # generate our list of possible nodes
    total_not_found = 0
    layers = set()
    for x in range(20, 141, 20):
        for y in range(20, 141, 20):
            for z in range(20, 141, 20):
                if not path_exists(x, y, z):  # only iterate if we have not done it yet
                    total_not_found += 1
                    layers.add((x, y, z))

    with open("results/incomplete_data.txt", 'w') as output_stream:
        output_stream.write("This is a list of node combinations that have not been complete\n")
        for item in layers:
            output_stream.write(f"To Do: {item}\n")
        output_stream.write(f"Total node combinations not found: {total_not_found}\n")

    # create a 'pool' of all possible workers
    pool = mp.Pool(mp.cpu_count())

    # iterate through our possible nodes
    start_time = time.time()
    for iteration in layers:
        # send a worker to complete the path
        pool.apply_async(multi_process_network, args=(iteration, nba_train_features, nba_train_labels, nba_val_features, nba_val_labels, nba_test_features))

    # bring workers back together
    pool.close()
    pool.join()
    end_time = time.time()
    with open("results/incomplete_data.txt") as output_stream:
        output_stream.write("")
        output_stream.write(f"Total time: {end_time - start_time}")
