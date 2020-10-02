import tensorflow as tf
from tensorflow import keras
import sys


class RegularNetwork(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(28, 28))
        self.hidden1 = keras.layers.Dense(200, activation='relu')
        self.hidden2 = keras.layers.Dense(100, activation='relu')
        self.hidden3 = keras.layers.Dense(60, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')

    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        hidden1 = self.hidden1(input_layer)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        output_layer = self.output_layer(hidden3)
        return output_layer


class DropoutNetwork(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(28, 28))
        self.hidden1 = keras.layers.Dense(200, activation='relu')
        self.hidden2 = keras.layers.Dense(100, activation='relu')
        self.hidden3 = keras.layers.Dense(60, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')
        self.dropout_layer = keras.layers.Dropout(rate=0.2)

    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        input_layer = self.dropout_layer(input_layer)
        hidden1 = self.hidden1(input_layer)
        hidden1 = self.dropout_layer(hidden1, training=training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.dropout_layer(hidden2, training=training)
        hidden3 = self.hidden3(hidden2)
        hidden3 = self.dropout_layer(hidden3, training=training)
        output_layer = self.output_layer(hidden3)
        return output_layer


def evaluate(input_model, train_images, train_labels, validation_images, validation_labels, epoch):
    sgd = keras.optimizers.SGD(lr=0.01)
    input_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    input_model.fit(train_images, train_labels, epochs=epoch, validation_data=(validation_images, validation_labels))

    loss, accuracy = input_model.evaluate(test_images, test_labels)
    return loss, accuracy


def dropout(data: list, epoch):
    model = DropoutNetwork()
    return evaluate(model, data[0], data[1], data[2], data[3], epoch)


def regular(data: list, epoch):
    model = RegularNetwork()
    return evaluate(model, data[0], data[1], data[2], data[3], epoch)


if __name__ == '__main__':
    # set up a default epoch to be used
    args = sys.argv
    try:
        epoch_value = int(args[1])
    except IndexError:
        epoch_value = 60  # default value of 60

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    validation_images = train_images[:5000]
    validation_labels = train_labels[:5000]

    data = [train_images, train_labels, validation_images, validation_labels]

    dropout_loss, dropout_accuracy = dropout(data, epoch_value)
    reg_loss, reg_accuracy = regular(data, epoch_value)

    print("\n\n")
    print("Dropout")
    print(f"Loss: {dropout_loss}")
    print(f"Accuracy: {dropout_accuracy}")

    print("\nRegular")
    print(f"Loss: {reg_loss}")
    print(f"Accuracy: {reg_accuracy}")
