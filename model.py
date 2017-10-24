import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import losses
import tensorflow as tf

from generate import *

def proxy_bce(y_true, y_pred):
    """_epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1-_epsilon)
    output = tf.log(output/(1-output))
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=output), axis=-1)"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def run(seq_length, num_examples, epochs=10, num_constraints=0):
    [inputs, outputs] = generate_pairwise_dependent(seq_length, num_examples, num_constraints)
    [train, test] = separate_train_test(inputs, outputs)
    model = Sequential()
    model.add(Dense(seq_length, activation='relu', input_dim=seq_length))
    model.add(Dense(seq_length, activation='sigmoid', input_dim=seq_length))
    model.compile(optimizer='sgd', loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(train[0], train[1], epochs=1000, batch_size=32)
    score = model.evaluate(test[0], test[1], batch_size=128)
    return score
