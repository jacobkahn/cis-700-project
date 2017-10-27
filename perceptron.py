import numpy as np
import itertools
import warnings

# Perceptron
class Constraint(object):
    pass

class Perceptron(object):
    def __init__(self, num_iterations, dataX, dataY):
        # Collections of constraints, manually added
        self.constraints = []
        self.num_iterations = num_iterations
        # Input features
        self.dataX = dataX
        self.dataY = dataY
        self.sanitize_input


    def sanitize_input(self):
        # Ensure we have as many in vectors as we do outputs
        if len(dataX) == 0 or len(dataY) == 0:
            raise Exception("Data of length zero given")
        if not len(dataX) == len(dataY):
            raise Exception("DataX and DataY not same length")
        if not len(dataX[0]) < 1:
            raise Exception("Sequence length zero for dataX")


    def feature_vector(self, x, y):
        """
        Computes a kernel function/feature vector for input x
        and structured output y.
        """
        if len(x) != len(y):
            return None
        n = len(y)
        vec = np.zeros(2*n)
        for i in range(n):
            if y[i] == 0:
                for i in range(n):
                    vec[i] += x[i]
            else:
                for i in range(n):
                    vec[i+n] += x[i]
        return vec


    def add_constraints(self, constraint):
        """
        Adds a constraint evaluated for inference
        """
        self.constraints.append(constraint)


    def _binary_arrs(self, length):
        """
        Return all lists of size length and with only 0's and 1's
        """
        strs = ["".join(seq) for seq in itertools.product("01", repeat=length)]
        arrs = []
        for s in strs:
            arrs.append([int(c) for c in s])
        return arrs


    def inference(self, x, w, constraints):
        """
        Return best y for given x, weight vector w, constraints, and
        feature vector function phi. Naively searches through each
        possible structured output.
        """
        candidates = self._binary_arrs(len(x))
        max_score = 0
        y = None
        for c in candidates:
            feature_vector = self.feature_vector(x, c)
            if np.dot(w, feature_vector) > max_score:
                valid_flag = True
                for const in constraints:
                    if not const.evaluate(c):
                        valid_flag = False
                        break
                if valid_flag:
                    y = c
        return y


    def run(self):
        """
        Runs the
        """
        # Set weight vectors
        self.w = np.zeros(len(self.dataX[0]))
        self.w_avg = np.zeros(len(self.dataX[0]))
        # counts the number of times we updated w_avg (we have a valid y prediction)
        iter_count = 0
        # Iterate as many times
        for iterNum in range(0, self.num_iterations):
            # Loop through x and y
            for x, y in zip(self.dataX, self.dataY):
                # Perform inference
                y_hat = self.inference(x, self.w, self.constraints)
                # Update weight vector
                if y_hat is not None:
                    self.w += self.feature_vector(x, y)-self.feature_vector(x, y_hat)
                    # Update average weight vector
                    self.w_avg = self.w_avg + self.w
                    iter_count += 1
        # Returns w_avg / (Tl): average weight vector divided by
        # number of tokens in the sequence x number of points
        return self.w_avg /iter_count



class ExampleConstraint(Constraint):
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
        self.index_arry = index_array
        self.assignments = poss_assign


    def evaluate(self, y):
        """
        evaluate the constraint on a specified data point y
        """
        const_projection = np.zeros(len(self.index_array))
        for i in range(len(self.index_array)):
            const_projection[i] = c[self.index_array[i]]
        return const_projection in self.assignments


someYData = [[]]
someXData = [[]]
structured_perceptron = Perceptron(1000, someXData, someYData)
structured_perceptron.add_constraints(ExampleConstraint([0, 1], [1, 1]))
result = structured_perceptron.run()
