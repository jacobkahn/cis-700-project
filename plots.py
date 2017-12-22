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


def multi_learning_curve_csv(csv_perc="perceptron_results.csv", csv_ffn="dnn_results.csv", csv_rnn="rnn_results.csv", csv_local="localclassifier_results.csv", noise=False, soft=False, seq_length=10):
    data = {'perceptron': pd.read_csv(csv_perc), 'rnn': pd.read_csv(csv_rnn), 'ffn': pd.read_csv(csv_ffn), 'localclassifier': pd.read_csv(csv_local)}
    for key in data.keys():
        data[key] = data[key][(data[key].soft == soft) & (data[key].noise == noise) & (data[key].seq_length == seq_length)]
    examples = range(100, 2100, 100)
    constraints = sorted(data['perceptron']['num_constraints'].unique())
    print(data['rnn'][data['rnn'].num_constraints == 0])
    graphdata = {c: {key: np.array([data[key][(data[key].num_examples == e) & (data[key].num_constraints == c)][key].iloc[0] for e in examples]) for key in data.keys()} for c in constraints}
    figs = range(111, 111+len(constraints))
    labels = {'perceptron': 'Perceptron', 'ffn': 'Feedforward Network', 'localclassifier': 'Local', 'rnn': 'RNN'}
    for c, f in zip(constraints, figs):
        fig = plt.figure(f)
        for key in data.keys():
            print(graphdata[c][key])
            print(len(graphdata[c][key]))
            plt.plot(examples, graphdata[c][key], label=labels[key])
        plt.legend()
        plt.xlim(examples[0], examples[-1])
        plt.ylim(0.5, 1)
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Test Set Accuracy')
        plt.title(str(c) + " constraints: Learning Curve")
        plt.show(block=False)

# if __name__ == "__main__":
    # learning_curve_csv("soft_results.csv")
