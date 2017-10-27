import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import losses
import tensorflow as tf
import itertools

def test_constraint(y1, y2):
    # Write constraint function - add constraints as bool exprs
    # returns true if should pass constraint
    return (y1 * y2 == 1)

def generate_subsets(seq_length, num_subsets, subset_size=2):
    subsets = set()
    c = 0
    while c < num_subsets:
        subset = np.random.choice(range(seq_length), subset_size, replace=False)
        if frozenset(subset) not in subsets:
            subsets.add(frozenset(subset))
            c += 1
    return subsets

def generate_independent(seq_length, num_examples):
    inputs = []
    for _ in range(num_examples):
        inputs.append(np.random.random_sample((seq_length,))*2-1)
    weights = []
    for _ in range(num_examples):
        weights.append(np.random.random_sample((seq_length,))*2-1)
    outputs = []
    for j in range(num_examples):
        output = []
        for i in range(seq_length):
            if (np.dot(weights[i], inputs[j]) > 0):
                output.append(1)
            else:
                output.append(0)
        outputs.append(np.array(output))
    return np.array(inputs), np.array(outputs)

def _binary_arrs(length):
    strs = ["".join(seq) for seq in itertools.product("01", repeat=length)]
    arrs = []
    for s in strs:
        arrs.append([int(c) for c in s])
    return arrs

def generate_pairwise_dependent(seq_length, num_examples, num_constraints):
    inputs = []
    for _ in range(num_examples):
        inputs.append(np.random.random_sample((seq_length,))*2-1)
    weights = []
    for _ in range(num_examples):
        weights.append(np.random.random_sample((seq_length,))*2-1)
    constraints = generate_subsets(seq_length, num_constraints, 2)
    print(constraints)
    outputs = []
    count = 0
    # for each example compute the argmax over constraint space
    for j in range(num_examples):
        f_vec = []
        for i in range(seq_length):
            f_vec.append(np.dot(weights[i], inputs[j]))
        f_vec = np.array(f_vec)
        y = None
        max_score = -np.inf
        binary_arrs = _binary_arrs(seq_length)
        # go through all structure
        for a in binary_arrs:
            # TODO: viterbi
            # check if cosntraint is violated. also only consider
            # constraints that beat the max score
            if np.dot(2*np.array(a)-1, f_vec) > max_score:
                invalid_flag = False
                for c in constraints:
                    c = list(c)
                    # constraint line
                    if test_constraint(a[c[0]], a[c[1]]):
                        invalid_flag = True
                        break
                    # don't consider if violated
                if not invalid_flag:
                    y = np.array(a)
                    max_score = np.dot(2*np.array(a)-1, f_vec)
        y_unconstrained = np.zeros((seq_length))
        for i in range(seq_length):
            if f_vec[i] >= 0:
                y_unconstrained[i] = 1
        # y is the argmax over the constraint space
        # y_unconstrainted is "" over unconstrained space
        if np.dot(y - y_unconstrained, y - y_unconstrained) != 0:
            count += 1
        outputs.append(y)
    print('Kappa:', float(count)/num_examples)
    return np.array(inputs), np.array(outputs)


def separate_train_test(inputs, outputs, test_frac=0.2):
    n = len(inputs)
    train_indices = np.random.choice(range(n), size=(int(0.8*n),), replace=False)
    test_indices = list(set(range(n))-set(train_indices))
    return [inputs[train_indices], outputs[train_indices]], [inputs[test_indices], outputs[test_indices]]
