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


def DropoutNetwork(input_data):
    model = keras.Sequential([
        # input layer; flatten will make a 2D array [[1, 2], [3, 4]] into a 1D array [1, 2, 3, 4]
        keras.layers.Flatten(input_shape=(input_data.shape[1], input_data.shape[2])),

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
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # convert data set to be between 0 - 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # set aside validation images
    validation_images = train_images[:5000]
    validation_labels = train_labels[:5000]

    # create a dropout network
    model = DropoutNetwork(train_images)

    # fit our model to data
    history = model.fit(train_images, train_labels, epochs=100, validation_data=(validation_images, validation_labels), batch_size=32)

    # evaluate model with testing data
    loss, accuracy = model.evaluate(test_images, test_labels)

    # view a history of data
    plot_history(history)
