import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import losses
import tensorflow as tf
import perceptron
from generate import *

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
        structured_perceptron = perceptron.Perceptron(20, self.seq_length)
        for constraint in constraints:
            structured_perceptron.add_constraints(constraint)
        train_result = structured_perceptron.train(self.train[0], self.train[1])
        test_result = structured_perceptron.test(self.test[0], self.test[1])
        return (train_result, test_result)



def proxy_bce(y_true, y_pred):
    """_epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1-_epsilon)
    output = tf.log(output/(1-output))
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=output), axis=-1)"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def run(seq_length, num_examples, epochs=10, num_constraints=0):
    [inputs, outputs, constraints] = generate_pairwise_dependent(seq_length, num_examples, num_constraints)
    [train, test] = separate_train_test(inputs, outputs)

    dlresult = DeepLearningClient(train, test, seq_length).run()
    perceptron_client = PerceptronClient(train, test, seq_length)
    [p_train, p_test] = perceptron_client.run(constraints)

    return (dlresult, p_test)


# the main function
if __name__ == "__main__":
    print run(10, 1000, num_constraints=2)
