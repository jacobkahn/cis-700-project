import matplotlib.pyplot as plt
from model import *
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def generate_learning_curve(seq_length, mutual_complexity):
    local = []
    ffn = []
    perc = []
    n_list = range(1000, 11000, 1000)
    for n in n_list:
        results = run(seq_length, n, mutual_complexity=mutual_complexity)
        local.append(results['local'])
        ffn.append(results['ffn'])
        perc.append(results['perceptron'])
    plt.plot(n_list, local, 'bo')
    plt.plot(n_list, ffn, 'ro')
    plt.plot(n_list, perc, 'go')
    plt.show()

# def learning_curve_csv(csv_name):
#     data = pd.read_csv(csv_name)
#     for c, f in zip([10, 20], [110, 111]):
#         print "Here with"
#         print (c, f)
#         fig = plt.figure(f)
#         plt.plot(data['mutual_complexity'], data['ffn'], 'b-', label='Feedforward Network')
#         plt.plot(data['mutual_complexity'], data['perceptron'], '-', color='violet', label='Structured Perceptron')
#         plt.plot(data['mutual_complexity'], data['localclassifier'], '-', color='orange', label='Local')
#         plt.legend()
#         plt.xlim(0, 30)
#         plt.ylim(0.5, 1)
#         plt.xlabel("Number of Constraints")
#         plt.ylabel("Test Set Accuracy")
#         plt.title("Accuracy vs. Number of Constraints - " + str(c) + " Seq Length")
#     plt.show(block=True)

def learning_curve_csv(csv_name):
    data = pd.read_csv(csv_name)
    # for c, f in zip([10, 20], [110, 111]):
    print "Here with"
    # print (c, f)
    fig = plt.figure(110)
    plt.plot(data['mutual_complexity'], data['ffn'], 'b-', label='Feedforward Network')
    plt.plot(data['mutual_complexity'], data['perceptron'], '-', color='violet', label='Structured Perceptron')
    plt.plot(data['mutual_complexity'], data['localclassifier'], '-', color='orange', label='Local')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0.5, 1)
    plt.xlabel("Number of Constraints")
    plt.ylabel("Test Set Accuracy")
    plt.title("Accuracy vs. Mutual Constraint Complexity")
    plt.show(block=True)


if __name__ == "__main__":
    learning_curve_csv("soft_results.csv")