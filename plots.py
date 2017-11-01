import matplotlib.pyplot as plt
from model import *

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
