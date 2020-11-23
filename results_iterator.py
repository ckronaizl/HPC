"""
This file will be used to iterate through the results/ folder and compare history of results

How do we want to compare the history data and predictions?
Methods:
    1) Only look at final value of history
        a) If current value is larger than next value, keep current and move to next
    2) Average of data over last 10 epochs
        a) Same as above: if current value larger than next value, keep current and move to next
    3) Compare network prediction results to actual
        a) If network ability to predict is better than next one, keep current and move on
    4)
"""
import os
import shutil
import pickle
import pandas as pd

root_dir = os.path.join(os.getcwd(), "results")

max_val_accuracy = int()
max_val_accuracy_path = str()

prediction_correct = int()
prediction_incorrect = int()
prediction_path = str()


def get_max_accuracy(history_obj, path):
    """
    This function compares the global validation accuracy with accuracy from history_obj
    If the history_obj validation accuracy is greater than the global value, the global value is replaced
    :param history_obj:
    :param path:
    :return:
    """
    # dont need these objects for now
    # testing_accuracy = history_obj["accuracy"]
    # testing_loss = history_obj["loss"]
    # validate_loss = history_obj["val_loss"]

    # we need access to these variables inside the funciton
    global max_val_accuracy
    global max_val_accuracy_path

    validate_accuracy = history_obj["val_accuracy"]
    if validate_accuracy[-1] > max_val_accuracy:
        max_val_accuracy = validate_accuracy[-1]
        max_val_accuracy_path = path


def get_highest_prediction(prediction_obj, path):
    """
    This functions compares the global prediction success rate with values from prediction_obj
    If the value from the current prediction_obj is greater than that of the global value, it is replaced
    :param prediction_obj:
    :param path:
    :return:
    """
    # make these variables available to function
    global prediction_correct
    global prediction_incorrect
    global prediction_path

    local_prediction_correct = int()
    local_prediction_incorrect = int()
    prediction_csv = pd.read_csv("test_train_data/NBA_test_15-20.csv", header=None,
                                 names=["W/L", "MIN", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
                                        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "+/-", "POSS", "TS%",
                                        "OER"])

    # iterate through our predictions and sum the correct/incorrect
    for i in range(prediction_obj.size):
        known_value = prediction_csv["W/L"][i]
        prediction_value = float(prediction_obj[i])

        # greater than 0.5 rounds up, equal to or less than 0.5 rounds down
        prediction_value = round(prediction_value, 0)

        # sum our correct and incorrect
        if prediction_value == known_value:
            local_prediction_correct += 1
        else:
            local_prediction_incorrect += 1

    # if our percentage correct is better than the global percentage correct, set new values
    try:
        global_percentage_correct = prediction_correct / (prediction_correct + prediction_incorrect)
    except ZeroDivisionError:
        global_percentage_correct = 0
    local_percentage_correct = local_prediction_correct / (local_prediction_correct + local_prediction_incorrect)
    if local_percentage_correct > global_percentage_correct:
        prediction_correct = local_prediction_correct
        prediction_incorrect = local_prediction_incorrect
        prediction_path = path


def check_data():
    """
    This will iterate through the files in root_dir and pull the history.pkl and predictions.pkl files
    It will then send each file through the appropriate function to determine which number of nodes is most optimal

    Current stats
    -------------
    Maximum validation accuracy: 85.3%
    Nodes to use for this accuracy: 120, 80, 140

    Maximum number of correct predictions: 24539
    Minimum number of incorrect predictions: 23
    Prediction rate: 99.9% correct
    Nodes to use for these values: 80, 80, 120

    :return:
    """
    global max_val_accuracy
    global max_val_accuracy_path
    global prediction_correct
    global prediction_incorrect
    global prediction_path

    for root, subdir, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "history.pkl" in file:
                history = pickle.load(open(file_path, "rb"))
                get_max_accuracy(history, file_path)
            elif "predictions.pkl" in file:
                predictions = pickle.load(open(file_path, "rb"))
                get_highest_prediction(predictions, file_path)
            else:
                continue

    with open("results/results_iterator.txt", 'w') as output_stream:
        output_stream.write(f"Maximum accuracy achieved: {max_val_accuracy}\n")
        output_stream.write(f"Path to data: {max_val_accuracy_path}\n")
        output_stream.write("")
        output_stream.write(f"Max correct predictions achieved: {prediction_correct}\n")
        output_stream.write(f"Minimum incorrect predictions: {prediction_incorrect}\n")
        output_stream.write(f"Path to data: {prediction_path}\n")


if __name__ == '__main__':
    generate_graph()
    check_data()
