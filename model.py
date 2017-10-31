import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import losses
import tensorflow as tf
import perceptron
from generate import *
import localclassifier

"""
Where the learning happens
"""
class LearningClient(object):
    def __init__(self, train, test, seq_length):
        self.train = train
        self.test = test
        self.seq_length = seq_length


class DeepLearningClient(LearningClient):
    def run(self):
        # do a deep learning
        model = Sequential()
        model.add(Dense(self.seq_length, activation='relu', input_dim=self.seq_length))
        model.add(Dense(self.seq_length, activation='sigmoid', input_dim=self.seq_length))
        model.compile(optimizer='sgd', loss="binary_crossentropy", metrics=['accuracy'])
        model.fit(self.train[0], self.train[1], epochs=1000, batch_size=32)
        score = model.evaluate(self.test[0], self.test[1], batch_size=128)
        return score


class PerceptronClient(LearningClient):
    def run(self, constraints):
        # do a structured_perceptron
        NUM_ITERS = 20
        structured_perceptron = perceptron.Perceptron(NUM_ITERS, self.seq_length)
        for constraint in constraints:
            structured_perceptron.add_constraints(constraint)
        train_result = structured_perceptron.train(self.train[0], self.train[1])
        test_result = structured_perceptron.test(self.test[0], self.test[1])
        return (train_result, test_result)


class LocalClassifierClient(LearningClient):
    def run(self):
        # Perceptrons for each digit
        localresults = []
        # For each element in sequence, create a perceptron
        NUM_ITERS = 20
        for i in range(self.seq_length):
            p = localclassifier.LocalClassifier(NUM_ITERS, self.seq_length)
            train_result = p.train(self.train[0], self.train[1], i)
            test_result = p.test(self.test[0], self.test[1], i)
            localresults.append((train_result, test_result))
        # sum losses over all spots
        totalerror = 0.0
        for i in range(self.seq_length):
            totalerror += localresults[i][1]
        # compute average error over all elements in sequence
        return round(float(totalerror) / (self.seq_length), 5)


def proxy_bce(y_true, y_pred):
    """_epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1-_epsilon)
    output = tf.log(output/(1-output))
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=output), axis=-1)"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def run(seq_length, num_examples, epochs=10, num_constraints=0):
    # Generate Data
    [inputs, outputs, constraints] = generate_pairwise_dependent(seq_length, num_examples, num_constraints)
    [train, test] = separate_train_test(inputs, outputs)

    # Naive classifier
    lcResult = -1
    lcResult = LocalClassifierClient(train, test, seq_length).run()

    # Deep learning
    dlresult = -1
    # dlresult = DeepLearningClient(train, test, seq_length).run()

    # Structured perceptron
    p_test = -1
    # perceptron_client = PerceptronClient(train, test, seq_length)
    # [p_train, p_test] = perceptron_client.run(constraints)

    return (lcResult, dlresult, p_test)


# the main function
if __name__ == "__main__":
    results = run(5, 1000, num_constraints=2)
    print "-------------------------------------------------------"
    print "RESULTS"
    print results
