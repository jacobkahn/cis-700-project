import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

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

def run(seq_length, num_examples, epochs=10):
    [inputs, outputs] = generate_independent(seq_length, num_examples)
    [train, test] = separate_train_test(inputs, outputs)
    model = Sequential()
    model.add(Dense(seq_length, activation='relu', input_dim=seq_length))
    model.add(Dense(seq_length, activation='sigmoid', input_dim=seq_length))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train[0], train[1], epochs=epochs, batch_size=32)
    score = model.evaluate(test[0], test[1], batch_size=128)
    return score
