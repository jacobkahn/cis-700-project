import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import losses
import tensorflow as tf

def generate_independent(seq_length, num_examples):
    inputs = []
    for _ in range(num_examples):
        inputs.append(np.random.random_sample((seq_length,))*2-1)
    weights = []
    for _ in range(num_examples):
        weights.append(np.random.random_sample((seq_length,))*10)
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

def separate_train_test(inputs, outputs, test_frac=0.2):
    n = len(inputs)
    train_indices = np.random.choice(range(n), size=(int(0.8*n),), replace=False)
    test_indices = list(set(range(n))-set(train_indices))
    return [inputs[train_indices], outputs[train_indices]], [inputs[test_indices], outputs[test_indices]]

def proxy_bce(y_true, y_pred):
    """_epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1-_epsilon)
    output = tf.log(output/(1-output))
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=output), axis=-1)"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def run(seq_length, num_examples, epochs=10):
    [inputs, outputs] = generate_independent(seq_length, num_examples)
    [train, test] = separate_train_test(inputs, outputs)
    model = Sequential()
    model.add(Dense(seq_length, activation='relu', input_dim=seq_length))
    model.add(Dense(seq_length, activation='sigmoid', input_dim=seq_length))
    model.compile(optimizer='sgd', loss=proxy_bce, metrics=['accuracy'])
    model.fit(train[0], train[1], epochs=1000, batch_size=32)
    score = model.evaluate(test[0], test[1], batch_size=128)
    return score
