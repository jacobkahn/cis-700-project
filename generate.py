import numpy as np
from perceptron import Constraint
import itertools
from gurobipy import *

class RandomConstraint(Constraint):
    """
    Sample constriant that checks
    """
    def __init__(self, index_array, poss_assign):
        """
        index_array: an array of indices for which the constraints are relevant
                     in the input y data
        poss_assign: the possible valid structured outputs for some function on
                     the data
        """
        self.index_array = list(index_array)
        self.assignments = poss_assign


    def evaluate(self, y):
        """
        evaluate the constraint on a specified data point y
        """
        const_projection = np.zeros(len(self.index_array))
        for i in range(len(self.index_array)):
            const_projection[i] = y[self.index_array[i]]
        for a in self.assignments:
            equal_flag = True
            for i in range(len(const_projection)):
                if const_projection[i] != a[i]:
                    equal_flag = False
                    break
            if equal_flag:
                return True
        return False

class GeneralConstraint(Constraint):
    def __init__(self, coeff, val):
        self.coeff = coeff
        self.val = val


def generate_subsets(seq_length, num_subsets, subset_size=2):
    subsets = set()
    c = 0
    while c < num_subsets:
        subset = np.random.choice(range(seq_length), subset_size, replace=False)
        if frozenset(subset) not in subsets:
            subsets.add(frozenset(subset))
            c += 1
    return subsets

def generate_coefficients(seq_length, num_constraints):
    arrs = []
    subsets = set()
    c = 0
    while c < num_constraints:
        size = np.random.choice(range(2, seq_length+1), 1)[0]
        subset = list(generate_subsets(seq_length, 1, subset_size=size))[0]
        if frozenset(subset) not in subsets:
            subsets.add(frozenset(subset))
            vec = np.zeros((seq_length))
            for i in range(seq_length):
                if i in subset:
                    vec[i] = 1
            arrs.append(vec)
            c += 1
    return arrs

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
    constraints = []
    constraints = generate_subsets(seq_length, num_constraints, 2)
    print(constraints)
    # constraints compatible with perceptron implementation
    good_constraints = []
    for constraint in constraints:
        good_constraints.append(RandomConstraint(constraint, [[0, 0], [0, 1],[1, 0]]))

    outputs = []
    count = 0
    for j in range(num_examples):
        f_vec = []
        for i in range(seq_length):
            f_vec.append(np.dot(weights[i], inputs[j]))
        f_vec = np.array(f_vec)
        y = None
        max_score = -np.inf
        binary_arrs = _binary_arrs(seq_length)
        for a in binary_arrs:
            if np.dot(2*np.array(a)-1, f_vec) > max_score:
                invalid_flag = False
                for c in constraints:
                    c = list(c)
                    if a[c[0]]*a[c[1]] == 1:
                        invalid_flag = True
                        break
                if not invalid_flag:
                    y = np.array(a)
                    max_score = np.dot(2*np.array(a)-1, f_vec)
        y_unconstrained = np.zeros((seq_length))
        for i in range(seq_length):
            if f_vec[i] >= 0:
                y_unconstrained[i] = 1
        if np.dot(y-y_unconstrained, y-y_unconstrained) != 0:
            count += 1
        outputs.append(y)
    print('Kappa:', float(count)/num_examples)
    return np.array(inputs), np.array(outputs), good_constraints

def generate_general(seq_length, num_training_examples, num_constraints, soft=False, noise=False):
    inputs = []
    num_examples = num_training_examples+100
    for _ in range(num_examples):
        inputs.append(np.random.random_sample((seq_length,))*2-1)
    weights = []
    for _ in range(num_examples):
        weights.append(np.random.random_sample((seq_length,))*2-1)
    constraints = []
    constraints = generate_coefficients(seq_length, num_constraints)
    print("Constraints:")
    print(constraints)
    # compute constraint complexity
    constraintcomplexity = 1
    # multiplicatively build up the constraint size
    for constraint in constraints:
        constraintcomplexity = constraintcomplexity + np.log((2 ** int(np.sum(constraint) / 2) + 1))
    logconstraintcomplexity = constraintcomplexity
    print("Constraint complexity is")
    print(logconstraintcomplexity)
    # calculate constraint complexity given seq length
    mutualcomplexity = logconstraintcomplexity / seq_length
    print("Mutual complexity is")
    print(mutualcomplexity)

    vals = []
    for i in range(len(constraints)):
        # vals.append(np.random.choice(range(1, int(np.sum(constraints[i]))), 1)[0])
        vals.append(int(np.sum(constraints[i]))/2)
    print(vals)
    good_constraints = []
    for constr, val in zip(constraints, vals):
        good_constraints.append(GeneralConstraint(constr, val))

    outputs = []
    for x in inputs:
        m = Model("MIP")
        m.setParam('OutputFlag', False)
        m_vars = []
        noisy_weights = []
        for i in range(seq_length):
            if noise:
                noisy_weights.append(weights[i]+np.random.normal(0, 0.2, size=(seq_length)))
            else:
                noisy_weights.append(weights[i])
            m_vars.append(None)
            m_vars[i] = m.addVar(vtype=GRB.BINARY, name=str(i))
        def obj():
            res = 0
            for i in range(seq_length):
                res += 2*m_vars[i]*np.dot(noisy_weights[i], x)-1
            return res
        m.setObjective(obj(), GRB.MAXIMIZE)
        for const in good_constraints:
            # General constraint handling
            if (not soft) or np.random.rand() < 0.5:
                m.addConstr(quicksum([const.coeff[i]*m_vars[i] for i in range(len(m_vars))]) <= const.val, str(const.coeff) + ' ' + str(const.val))
        m.optimize()
        y = []
        for i in range(seq_length):
            y.append(m_vars[i].x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs), good_constraints, logconstraintcomplexity, mutualcomplexity

def separate_train_test(inputs, outputs, test_size=100):
    n = len(inputs)
    train_indices = np.random.choice(range(n), size=(int(n-test_size),), replace=False)
    # train_indices = np.random.choice(range(n), size=(int((1-test_frac)*n),), replace=False)
    test_indices = list(set(range(n))-set(train_indices))
    return [inputs[train_indices], outputs[train_indices]], [inputs[test_indices], outputs[test_indices]]
