import numpy as np
import itertools
import warnings
from gurobipy import *

class Constraint(object):
    pass

class Perceptron(object):
    def __init__(self, num_iterations, seq_length):
        # Collections of constraints, manually added
        self.constraints = []
        self.num_iterations = num_iterations
        self.seq_length = seq_length
        self.wv_length = 2*seq_length**2

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
        vec = np.zeros(self.wv_length)
        for i in range(self.seq_length):
            if y[i] == 0:
                for j in range(self.seq_length):
                    vec[i*self.seq_length+j] += x[j]
            else:
                for j in range(self.seq_length):
                    vec[(i+1)*self.seq_length+j] += x[j]
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


    def inference(self, x, w, use_ilp=True):
        """
        Return best y for given x, weight vector w, constraints, and
        feature vector function phi. Naively searches through each
        possible structured output.
        """
        # Naive inference
        if not use_ilp:
            candidates = self._binary_arrs(self.seq_length)
            max_score = 0
            y = None
            for c in candidates:
                feature_vector = self.feature_vector(x, c)
                this_score = np.dot(w, feature_vector)
                if this_score > max_score or y is None:
                    valid_flag = True
                    for const in self.constraints:
                        if not const.evaluate(c):
                            valid_flag = False
                            break
                    if valid_flag:
                        y = c
                        max_score = this_score
            return y
        # ILP-based inference (currently supports only constraints of form
        # y_iy_j != 1)
        m = Model("MIP")
        m.setParam('OutputFlag', False)
        m_vars = []
        for i in range(self.seq_length):
            m_vars.append(None)
            m_vars[i] = m.addVar(vtype=GRB.BINARY, name=str(i))
        def obj():
            res = 0
            for i in range(self.seq_length):
                res += np.dot(w[i*self.seq_length:(i+1)*self.seq_length], x)*(1-m_vars[i])+np.dot(w[(i+1)*self.seq_length:(i+2)*self.seq_length], x)*m_vars[i]
            return res
        m.setObjective(obj(), GRB.MAXIMIZE)
        for const in self.constraints:
            i = const.index_array[0]
            j = const.index_array[1]
            m.addConstr(m_vars[i]+m_vars[j] <= 1, str(i) + ' ' + str(j))
        m.optimize()
        y = []
        for i in range(self.seq_length):
            y.append(m_vars[i].x)
        return y


    def loss_function(self, y, y_hat):
        """
        given a y_hat (prediction) and actual y, compute error
        """
        return np.sum(np.abs(np.array(y) - np.array(y_hat)))


    def train(self, dataX, dataY, use_ilp=True):
        """
        Runs the
        """
        # Input features
        self.sanitize_input(dataX, dataY)
        self.trainDataX = dataX
        self.trainDataY = dataY

        # Set weight vectors
        self.w = np.zeros(self.wv_length)
        self.w_avg = np.zeros(self.wv_length)
        # counts the number of times we updated w_avg (we have a valid y prediction)
        iter_count = 0
        # Iterate as many times
        for iterNum in range(0, self.num_iterations):
            print "Perceptron: iteration " + str(iter_count)
            # Loop through x and y
            for x, y in zip(self.trainDataX, self.trainDataY):
                # Perform inference
                y_hat = self.inference(x, self.w, use_ilp=use_ilp)
                # Update weight vector
                if y_hat is not None:
                    self.w += self.feature_vector(x, y)-self.feature_vector(x, y_hat)
                    # Update average weight vector
                    self.w_avg = self.w_avg + self.w
                    iter_count += 1
        # Returns w_avg / (Tl): average weight vector divided by
        # number of tokens in the sequence x number of points
        self.w_avg /= iter_count
        return self.w_avg


    def test(self, dataX, dataY, use_ilp=True):
        # Input features
        self.sanitize_input(dataX, dataY)
        self.testDataX = dataX
        self.testDataY = dataY

        self.total_error = 0
        for example, actual in zip(self.testDataX, self.testDataY):
            y_hat = self.inference(example, self.w_avg, use_ilp=use_ilp)
            self.total_error += self.loss_function(y_hat, actual)
        return float(self.total_error) / (len(self.testDataY) * self.seq_length)
