# initial article on dropout networks: https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
# current code from: https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/14-%20Solving%20overfitting%20in%20neural%20networks/code/solving_overfitting.py
# video on code at: https://www.youtube.com/watch?v=Gf5DO6br0ts&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=14
import tensorflow as tf
from tensorflow import keras
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import os
import time


def DropoutNetwork(input_data):
    model = keras.Sequential([
        # input layer; flatten will make a 2D array [[1, 2], [3, 4]] into a 1D array [1, 2, 3, 4]
        keras.layers.Flatten(input_shape=(input_data.shape[1], input_data.shape[2])),

        # first dense layer
        keras.layers.Dense(200, activation='relu'),
        # keras.layers.Dropout(0.3),

        # second dense layer
        keras.layers.Dense(100, activation='relu'),
        # keras.layers.Dropout(0.3),

        # third dense layer
        keras.layers.Dense(60, activation='relu'),
        # keras.layers.Dropout(0.3),

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
    fig, (ax1, ax2) = plt.subplots(2, sharex="all")

    # history is a dictionary
    # history.history has four values: accuracy, val_accuracy, loss, val_loss
    # create new plot
    ax1.plot(history.history["accuracy"], label="train accuracy")
    ax1.plot(history.history["val_accuracy"], label="test accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.set_title("Accuracy eval")

    # create error subplot
    ax2.plot(history.history["loss"], label="train error")
    ax2.plot(history.history["val_loss"], label="test error")
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right")
    ax2.set_title("Error eval")

    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/No Dropout Results.png")


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
