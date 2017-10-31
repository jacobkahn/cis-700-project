import numpy

class LocalClassifier(object):
    def __init__(self, num_iterations, seq_length):
        # Collections of constraints, manually added
        self.num_iterations = num_iterations
        self.seq_length = seq_length

    def sanitize_input(self, dataX, dataY):
        # Doesn't actually do the length comparisons fyi
        # change the 0 Y values to -1
        for i in range(len(dataY)):
            for j in range(self.seq_length):
                if dataY[i][j] == 0:
                    dataY[i][j] = -1
        return dataY

    def train(self, dataX, dataY, seq_el_idx):
        # index in the sequence to learn now
        self.seq_el_idx = seq_el_idx
        print "Local perceptron running for index " + str(self.seq_el_idx)
        # Input features, change 0 to -1
        dataY = self.sanitize_input(dataX, dataY)
        # extract points at index
        self.trainDataX = [point[self.seq_el_idx] for point in dataX]
        self.trainDataY = [point[self.seq_el_idx] for point in dataY]
        # weight for this index
        # self.w = np.zeros(self.wv_length)
        # self.w_avg = np.zeros(self.wv_length)
        self.w = 0
        for iterNum in range(0, self.num_iterations):
            for x, y in zip(self.trainDataX, self.trainDataY):
                y_hat = self.w * x
                if numpy.sign(y_hat) != numpy.sign(y):
                    self.w += x * y
        return self.w


    def test(self, dataX, dataY, seq_el_idx):
        # index in the sequence to test now
        self.seq_el_idx = seq_el_idx
        # Input features
        dataY = self.sanitize_input(dataX, dataY)
        self.testDataX = [point[self.seq_el_idx] for point in dataX]
        self.testDataY = [point[self.seq_el_idx] for point in dataY]

        self.total_error = 0
        for example, actual in zip(self.testDataX, self.testDataY):
            y_hat = self.w * example
            # loss function is just number of mispredicts
            self.total_error += int(numpy.sign(y_hat) != numpy.sign(actual))
        return float(self.total_error) / len(self.testDataY)
