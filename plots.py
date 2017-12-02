import matplotlib.pyplot as plt
from model import *
import pandas as pd

def generate_learning_curve(seq_length, num_constraints):
    local = []
    ffn = []
    perc = []
    n_list = range(1000, 11000, 1000)
    for n in n_list:
        results = run(seq_length, n, num_constraints=num_constraints)
        local.append(results['local'])
        ffn.append(results['ffn'])
        perc.append(results['perceptron'])
    plt.plot(n_list, local, 'bo')
    plt.plot(n_list, ffn, 'ro')
    plt.plot(n_list, perc, 'go')
    plt.show()

def learning_curve_csv(csv_name):
    data = pd.read_csv(csv_name)
    for c, f  in zip([0, 5, 10, 15], [111, 112, 113, 114]):
        plt.figure(f)
        plt.plot(data[data.constraints == c]['training_size'], data[data.constraints == c]['ffn'], 'b-', label='Feedforward Network')
        plt.plot(data[data.constraints == c]['training_size'], data[data.constraints == c]['perceptron'], '-', color='violet', label='Structured Perceptron')
        plt.plot(data[data.constraints == c]['training_size'], data[data.constraints == c]['local'], '-', color='orange', label='Local')
        plt.legend()
        plt.xlim(0, 2100)
        plt.ylim(0.5, 1)
        plt.xlabel("Training Set Size")
        plt.ylabel("Test Set Accuracy")
        plt.title("Accuracy vs. Training Set Size - " + str(c) + " Constraints")
    plt.show(block=False)
