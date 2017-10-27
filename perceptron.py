import numpy as np
import itertools
import warnings

class Constraint(object):
    pass

class Perceptron(object):
    def __init__(self, num_iterations, seq_length):
        # Collections of constraints, manually added
        self.constraints = []
        self.num_iterations = num_iterations
        self.seq_length = seq_length

    def sanitize_input(self, dataX, dataY):
        # Ensure we have as many in vectors as we do outputs
        if len(dataX) == 0 or len(dataY) == 0:
            raise Exception("Data of length zero given")
        if not len(dataX) == len(dataY):
            raise Exception("DataX and DataY not same length")
        if len(dataX[0]) < 1:
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


    def inference(self, x, w):
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
            if np.dot(w, feature_vector) > max_score or y is None:
                valid_flag = True
                for const in self.constraints:
                    if not const.evaluate(c):
                        valid_flag = False
                        break
                if valid_flag:
                    y = c
        return y


    def loss_function(self, y, y_hat):
        """
        given a y_hat (prediction) and actual y, compute error
        """
        return np.sum(np.abs(np.array(y) - np.array(y_hat)))


    def train(self, dataX, dataY):
        """
        Runs the
        """
        # Input features
        self.sanitize_input(dataX, dataY)
        self.trainDataX = dataX
        self.trainDataY = dataY

        # Set weight vectors
        self.w = np.zeros(2 * len(self.trainDataX[0]))
        self.w_avg = np.zeros(2 * len(self.trainDataX[0]))
        # counts the number of times we updated w_avg (we have a valid y prediction)
        iter_count = 0
        # Iterate as many times
        for iterNum in range(0, self.num_iterations):
            print "Perceptron: iteration " + str(iter_count)
            # Loop through x and y
            for x, y in zip(self.trainDataX, self.trainDataY):
                # Perform inference
                y_hat = self.inference(x, self.w)
                # Update weight vector
                if y_hat is not None:
                    self.w += self.feature_vector(x, y)-self.feature_vector(x, y_hat)
                    # Update average weight vector
                    self.w_avg = self.w_avg + self.w
                    iter_count += 1
        # Returns w_avg / (Tl): average weight vector divided by
        # number of tokens in the sequence x number of points
        return self.w_avg / iter_count


    def test(self, dataX, dataY):
        # Input features
        self.sanitize_input(dataX, dataY)
        self.testDataX = dataX
        self.testDataY = dataY

        self.total_error = 0
        for example, actual in zip(self.testDataX, self.testDataY):
            y_hat = self.inference(example, self.w_avg)
            self.total_error += self.loss_function(y_hat, actual)
        return float(self.total_error) / (len(self.testDataY) * self.seq_length)
