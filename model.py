import numpy as np
import tensorflow as tf
import perceptron
from generate import *
import localclassifier
from functools import partial
import multiprocessing

"""
Where the learning happens
"""
class LearningClient(object):
    def __init__(self, train, test, seq_length):
        self.train = train
        self.test = test
        self.seq_length = seq_length

    def get_identifying_key(self):
        raise Exception('Unimplemented in base: LearningClient.get_identifying_key')

    def run(self):
        raise Exception('Unimplemented in base: LearningClient.run')

    def run_parallelized_compute(self, shared_memory):
        shared_memory[self.get_identifying_key()] = self.run()


class RNNClient(LearningClient):
    def get_identifying_key(self):
        return 'rnn'

    def run(self):
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Input, Concatenate, LSTM, RepeatVector
        from keras.optimizers import RMSprop, SGD, Adam
        from keras import backend as K
        from keras import losses
        # Reshape train data
        self.train[0] = np.array(self.train[0])
        newdata = []
        for i, item in enumerate(self.train[0]):
            new = []
            for i in range(0, self.seq_length):
                new.append(item)
            newdata.append(np.array(new))

        self.train[0] = np.array(newdata)


        # Reshape test data
        self.test[0] = np.array(self.test[0])
        newdata = []
        for i, item in enumerate(self.test[0]):
            new = []
            for i in range(0, self.seq_length):
                new.append(item)
            newdata.append(np.array(new))
        self.test[0] = np.array(newdata)

        HIDDEN_SIZE = self.seq_length
        model = Sequential()

        # model.add(LSTM(HIDDEN_SIZE, input_shape=(self.seq_length, 1)))
        model.add(LSTM(HIDDEN_SIZE, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(self.seq_length, self.seq_length)))
        # model.add(Dense(self.seq_length, activation='relu', input_dim=self.seq_length))

        def loss(y_true, y_pred, alpha=0.001):
            bce = K.binary_crossentropy(y_true, y_pred)
            return bce

        # model.add(Dense(self.seq_length, activation='sigmoid', input_dim=2*self.seq_length))
        # model.add(Dropout(0.2))


        sgd = SGD(lr=0.3, decay=0, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.0002, decay=0)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.train[0], self.train[1], epochs=300)
        # model.fit(self.train[0], self.train[1], epochs=900)
        score = model.evaluate(self.test[0], self.test[1], batch_size=128)
        return score[1]


class DeepLearningClient(LearningClient):
    def get_identifying_key(self):
        return 'ffn'

    def run(self):
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Input, Concatenate, LSTM, RepeatVector
        from keras.optimizers import RMSprop, SGD, Adam
        from keras import backend as K
        from keras import losses
        # do a deep learning
        model = Sequential()
        model.add(Dense(2*self.seq_length, activation='relu', input_dim=self.seq_length))
        # model.add(Dense(self.seq_length, activation='tanh', input_dim=self.seq_length))
        model.add(Dense(self.seq_length, activation='sigmoid', input_dim=2*self.seq_length))
        model.add(Dropout(0.2))
        sgd = SGD(lr=0.3, decay=0, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.0002, decay=0)
        """gan = GanClient(self.train, self.test, self.seq_length)
        gan.run()
        k_one = K.variable(value=1.0)
        weights1_arr = gan.discrim.layers[0].layers[0].get_weights()[0]
        weights1 = K.variable(value=weights1_arr)
        biases1_arr = gan.discrim.layers[0].layers[0].get_weights()[1]
        biases1 = K.variable(value=biases1_arr)
        weights2_arr = gan.discrim.layers[0].layers[1].get_weights()[0]
        weights2 = K.variable(value=weights2_arr)
        biases2_arr = gan.discrim.layers[0].layers[1].get_weights()[1]
        biases2 = K.variable(value=biases2_arr)
        weights3_arr = gan.discrim.layers[0].layers[2].get_weights()[0]
        weights3 = K.variable(value=weights3_arr)
        biases3_arr = gan.discrim.layers[0].layers[2].get_weights()[1]
        biases3 = K.variable(value=biases3_arr)
        def gan_predict(y):
            return K.sigmoid(K.dot(K.sigmoid(K.dot(K.relu(K.dot(y, weights1)+biases1), weights2)+biases2), weights3)+biases3)
        def np_sigmoid(x):
            return 1.0/(1.0+np.exp(-1*x))
        def gan_predict_arr(y):
            return np_sigmoid(np.dot(np_sigmoid(np.dot(np.maximum(0, np.dot(y, weights1_arr)+biases1_arr), weights2_arr)+biases2_arr), weights3_arr)+biases3_arr)
        print('prediction:')
        err = 0
        for y in self.test[1]:
            err += 1-gan_predict_arr(y)
        print(err)"""
        def loss(y_true, y_pred, alpha=0.001):
            bce = K.binary_crossentropy(y_true, y_pred)
            return bce
            # return bce+alpha*K.log(gan_predict(y_pred))
            # return K.switch(K.greater(K.variable(value=0.4), bce), bce+alpha*(k_one-K.sigmoid(K.dot(K.sigmoid(K.dot(K.relu(K.dot(y_pred, weights1)+biases1), weights2)+biases2), weights3)+biases3)), bce)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.train[0], self.train[1], epochs=300)
        # model.fit(self.train[0], self.train[1], epochs=900)
        score = model.evaluate(self.test[0], self.test[1], batch_size=128)
        return score[1]


class TFClient(LearningClient):
    def get_identifying_key(self):
        return 'tf'

    def run(self):
        graph = tf.Graph()
        # gan = GanClient(self.train, self.test, self.seq_length)
        # gan.run()
        with graph.as_default():
            tf_train_x = tf.placeholder(tf.float32, shape=[None, self.seq_length])
            tf_train_y = tf.placeholder(tf.float32, shape=[None, self.seq_length])
            tf_test_x = tf.constant(self.test[0])
            layer1_weights = tf.Variable(tf.truncated_normal([self.seq_length, self.seq_length]))
            layer1_biases = tf.Variable(tf.zeros([self.seq_length]))
            layer2_weights = tf.Variable(tf.truncated_normal([self.seq_length, self.seq_length]))
            layer2_biases = tf.Variable(tf.zeros([self.seq_length]))
            layer3_weights = tf.Variable(tf.truncated_normal([self.seq_length, self.seq_length]))
            layer3_biases = tf.Variable(tf.zeros([self.seq_length]))
            def three_layer_network(data):
                input_layer = tf.matmul(tf.cast(data, tf.float32), tf.cast(layer1_weights, tf.float32))
                hidden = tf.nn.relu(input_layer + layer1_biases)
                hidden2 = tf.nn.sigmoid(tf.matmul(hidden, layer2_weights)+layer2_biases)
                output_layer = tf.nn.sigmoid(tf.matmul(hidden2, layer3_weights) +layer3_biases)
                return output_layer
            model_scores = three_layer_network(tf_train_x)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_scores, labels=tf_train_y))
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
            train_prediction = model_scores
            test_prediction = three_layer_network(tf_test_x)
        def accuracy(predictions, actual):
            print(len(predictions), len(actual))
            return 1-np.sum(np.abs(np.array(predictions)-np.array(actual)))/(self.seq_length*len(predictions))
        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            num_steps = 20000
            batch_size = len(self.train[1])
            for step in range(num_steps):
                # offset = (step*batch_size) % (len(self.train[1]) - batch_size
                offset = 0
                minibatch_x = self.train[0][offset:(offset+batch_size),:]
                minibatch_y = self.train[1][offset:(offset+batch_size)]
                feed_dict = {tf_train_x: minibatch_x, tf_train_y: minibatch_y}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if step % 1000 == 0:
                    print('Minibatch loss at step {0}: {1}'.format(step, l))
            return accuracy(test_prediction.eval(), self.test[1])


class GanClient(LearningClient):
    def get_identifying_key(self):
        return 'gan'

    iterations = 6000
    discrim_dropout = 0.1
    gen_dropout = 0.1
    def discriminator(self):
        self.D = Sequential()
        self.D.add(Dense(self.seq_length, activation='relu', input_dim=self.seq_length))
        self.D.add(Dense(self.seq_length, activation='sigmoid', input_dim=self.seq_length))
        # self.D.add(Dropout(self.discrim_dropout))
        self.D.add(Dense(1, activation='sigmoid'))
        return self.D
    def generator(self):
        self.G = Sequential()
        self.G.add(Dense(self.seq_length, activation='relu', input_dim=self.seq_length))
        self.G.add(Dense(self.seq_length, activation='sigmoid', input_dim=self.seq_length))
        self.G.add(Dropout(self.gen_dropout))
        return self.G
    def discrim_model(self):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM
    def adversarial_model(self):
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM
    def run(self):
        self.gen = self.generator()
        self.discrim = self.discrim_model()
        self.adv = self.adversarial_model()
        for _ in range(self.iterations):
            noise = np.random.uniform(-1.0, 1.0, size=[len(self.train[0]), self.seq_length])
            artificial_ys = self.gen.predict(noise)
            x = np.concatenate((self.train[1], artificial_ys))
            y = np.ones([2*len(self.train[0]), 1])
            y[len(self.train[0]):, :] = 0
            d_loss = self.discrim.train_on_batch(x, y)
            y = np.ones([len(self.train[0]), 1])
            noise = np.random.uniform(-1.0, 1.0, size=[len(self.train[0]), self.seq_length])
            a_loss = self.adv.train_on_batch(noise, y)
        err = 0.0
        for i in range(len(self.test[0])):
            yi_hat = self.gen.predict(np.array([self.test[0][i]]))[0]
            err += np.dot(np.abs(np.array(yi_hat)-self.test[1][i]), np.abs(np.array(yi_hat)-self.test[1][i]))
        print(self.discrim.evaluate(self.test[1], np.ones((len(self.test[1]))), batch_size=128))
        score = (self.seq_length*len(self.test[0])-err)/(self.seq_length*len(self.test[0]))
        return score

class DanClient(LearningClient):
    def get_identifying_key(self):
        return 'dan'

    iterations = 300
    discrim_dropout = 0.7
    pred_dropout = 0.2
    alpha = 0.03
    def discriminator(self):
        self.Dlayers = []
        self.D = Sequential()
        # self.Dlayers.append(Dense(2*self.seq_length, activation='relu', input_dim=self.seq_length))
        self.Dlayers.append(Dense(3*self.seq_length, activation='relu', input_dim=self.seq_length))
        # self.Dlayers.append(Dropout(self.discrim_dropout))
        # self.Dlayers.append(keras.layers.LeakyReLU(alpha=self.alpha, input_shape=(2*self.seq_length,)))
        self.Dlayers.append(Dense(3*self.seq_length, activation='sigmoid', input_dim=3*self.seq_length))
        # self.Dlayers.append(keras.layers.LeakyReLU(alpha=self.alpha, input_shape=(self.seq_length/3,)))
        self.Dlayers.append(Dropout(self.discrim_dropout))
        self.Dlayers.append(Dense(1, activation='sigmoid', input_dim=3*self.seq_length))
        for l in self.Dlayers:
            self.D.add(l)
        """self.discrim_input = Input(shape=(self.seq_length,))
        self.Dlayers.append(Dense(1, activation='linear', input_dim=self.seq_length)(self.discrim_input))
        self.Dlayers.append(Dense(1, activation='linear', input_dim=self.seq_length)(self.discrim_input))
        self.Dlayers.append(keras.layers.maximum(self.Dlayers))
        # self.Dlayers.append(keras.layers.Lambda(lambda x: x[1]-x[0])([self.Dlayers[-1], self.Dlayers[-2]]))
        self.D = keras.models.Model(inputs=[self.discrim_input], outputs=self.Dlayers[-1])"""
        return self.D
    def predictor(self):
        # self.input1 = Input(shape=(self.seq_length,))
        # pred_layer1 = keras.layers.LeakyReLU(alpha=self.alpha)(self.input1)
        # pred_layer1 = Dense(self.seq_length, activation='relu', input_dim=self.seq_length)(self.input1)
        # pred_layer2 = Dense(self.seq_length, activation='sigmoid', input_dim=self.seq_length)(pred_layer1)
        # pred_layer3 = keras.layers.LeakyReLU(alpha=self.alpha)(pred_layer2)
        # pred_layer3 = keras.layers.BatchNormalization()(pred_layer1)
        # pred_layer4 = Dense(self.seq_length, activation='tanh', input_dim=self.seq_length)(pred_layer3)
        # pred_layer_out = Dropout(self.pred_dropout)(pred_layer2)
        # self.P = keras.models.Model(inputs=[self.input1], outputs=pred_layer2)
        self.P = Sequential()
        self.P.add(Dense(2*self.seq_length, activation='relu', input_dim=self.seq_length))
        self.P.add(Dense(self.seq_length, activation='sigmoid', input_dim=2*self.seq_length))
        self.P.add(Dropout(self.pred_dropout))
        self.P.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.P
    def discrim_model(self):
        optimizer = Adam(lr=0.0002, decay=0)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return self.DM
    def adversarial_model(self):
        optimizer = Adam(lr=0.0002, decay=0)
        for l in self.Dlayers:
            l.trainable = False
        self.D.trainable = False
        self.input2 = Input(shape=(self.seq_length,))
        self.input1 = Input(shape=(self.seq_length,))
        prediction = self.pred(self.input1)
        # self.merge_layer = Concatenate()([self.input2, prediction])
        """output_tensor = self.Dlayers[0](prediction)
        for l in range(1, len(self.Dlayers)):
            output_tensor = self.Dlayers[l](output_tensor)"""
        output_tensor = self.D(prediction)
        self.AM = keras.models.Model(inputs=[self.input1, self.input2], outputs=[output_tensor, prediction])
        self.AM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], loss_weights=[1., 1.])
        return self.AM
    def run(self):
        self.pred = self.predictor()
        self.discrim = self.discrim_model()
        self.adv = self.adversarial_model()
        for i in range(self.iterations):
            predicted_ys = self.pred.predict(self.train[0])
            for _ in range(3):
                # discrim_input = np.concatenate((self.train[0], predicted_ys), axis=1)
                discrim_input = predicted_ys
                discrim_output = np.zeros([len(self.train[0]), 1])

                d1_loss = self.discrim.fit(discrim_input, discrim_output, epochs=1)
                # print('D1:', d_loss)
                # discrim_input = np.concatenate((self.train[0], self.train[1]), axis=1)
                discrim_input = self.train[1]
                discrim_output = np.ones([len(self.train[0]), 1])
                d2_loss = self.discrim.fit(discrim_input, discrim_output, epochs=1)
                # print('D2:', d_loss)
                """discrim_input = np.concatenate((np.concatenate((self.train[0], predicted_ys), axis=1), np.concatenate((self.train[0], self.train[1]), axis=1)))
                discrim_output = np.zeros([2*len(self.train[0]), 1])
                discrim_output[len(self.train[0]):, :] = 1.0
                d_loss = self.discrim.train_on_batch(discrim_input, discrim_output)"""
            # print('D2:', d_loss)
            adv_output = [np.ones([len(self.train[0]), 1]), self.train[1]]
            a_loss = self.adv.fit([self.train[0], self.train[0]], adv_output, epochs=3)
            # print('A:', a_loss)
            y_hat = self.pred.predict(np.array(self.train[0]))
            diff = np.abs(np.array(y_hat)-np.array(self.train[1]))
            err = np.sum(diff)
            print(str(i), (self.seq_length*len(self.train[0])-err)/(self.seq_length*len(self.train[0])), d1_loss, d2_loss)
        y_hat = self.pred.predict(np.array(self.test[0]))
        diff = np.abs(np.array(y_hat)-np.array(self.test[1]))
        err = np.sum(diff)
        score = (self.seq_length*len(self.test[0])-err)/(self.seq_length*len(self.test[0]))
        return score


class GanConstraint(Constraint):
    def __init__(self, gan):
        self.gan = gan
    def evaluate(self, y):
        if self.gan.discrim.predict(np.array([y]))[0] == 1:
            return True
        return False


class PerceptronGanClient(LearningClient):
    def get_identifying_key(self):
        return 'ganperceptron'

    def run(self):
        NUM_ITERS = 20
        gan = GanClient(self.train, self.test, self.seq_length)
        gan.run()
        structured_perceptron = perceptron.Perceptron(NUM_ITERS, self.seq_length)
        structured_perceptron.add_constraints(GanConstraint(gan))
        train_result = structured_perceptron.train(self.train[0], self.train[1], use_ilp=False)
        test_result = structured_perceptron.test(self.test[0], self.test[1], use_ilp=False)
        return (train_result, 1-test_result)

class PerceptronClient(LearningClient):
    def get_identifying_key(self):
        return 'perceptron'

    def add_constraints(self, constraints):
        self.constraints = constraints

    def run(self):
        # do a structured_perceptron
        NUM_ITERS = 20
        structured_perceptron = perceptron.Perceptron(NUM_ITERS, self.seq_length)
        for constraint in self.constraints:
            structured_perceptron.add_constraints(constraint)
        train_result = structured_perceptron.train(self.train[0], self.train[1])
        test_result = structured_perceptron.test(self.test[0], self.test[1])
        # return (train_result, 1-test_result)
        return 1 - test_result


class LocalClassifierClient(LearningClient):
    def get_identifying_key(self):
        return 'localclassifier'

    def run(self):
        # Perceptrons for each digit
        localresults = []
        # For each element in sequence, create a perceptron
        NUM_ITERS = 10
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
        return round(1-float(totalerror) / (self.seq_length), 5)


def proxy_bce(y_true, y_pred, gan=None):
    """_epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1-_epsilon)
    output = tf.log(output/(1-output))
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=output), axis=-1)"""
    # return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    if gan is None:
        return losses.binary_crossentropy(y_true, y_pred)
    with tf.Session().as_default():
        return losses.binary_crossentropy(y_true, y_pred)+1-gan.discrim.predict(y_pred.eval())

def run_parallel_compute(obj, args):
    """
    shim that helps serialize an instance method for a new process hook
    """
    obj.run_parallelized_compute(args)

def run(seq_length, num_examples, num_constraints=0, soft=False, noise=False):
    # Set up shared data across processes
    manager = multiprocessing.Manager()
    # this is a shared map with mutex proclocks
    shared_results = manager.dict()
    # collection of process threads to be joined
    jobs = []

    # Generate Data
    [inputs, outputs, constraints, complexity, mutualcomplexity] = generate_general(seq_length, num_examples, num_constraints, soft, noise)
    [train, test] = separate_train_test(inputs, outputs)
    shared_results['constraint_complexity'] = complexity
    shared_results['mutual_complexity'] = mutualcomplexity

    shared_results['seq_length'] = seq_length
    shared_results['num_constraints'] = num_constraints
    shared_results['num_examples'] = num_examples
    shared_results['soft'] = soft
    shared_results['noise'] = noise

    # Naive classifier
    lc_acc = LocalClassifierClient(train, test, seq_length)
    p = multiprocessing.Process(target=run_parallel_compute, args=(lc_acc, shared_results))
    jobs.append(p)
    p.start()

    # Deep learning
    dl_acc = DeepLearningClient(train, test, seq_length)
    p = multiprocessing.Process(target=run_parallel_compute, args=(dl_acc, shared_results))
    jobs.append(p)
    p.start()

    # RNN
    rnn_acc = RNNClient(train, test, seq_length)
    p = multiprocessing.Process(target=run_parallel_compute, args=(rnn_acc, shared_results))
    jobs.append(p)
    p.start()

    # Tensorflow
    # tf_acc = TFClient(train, test, seq_length)
    # p = multiprocessing.Process(target=run_parallel_compute, args=(tf_acc, shared_results))
    # jobs.append(p)
    # p.start()


    # GAN
    # gan = PerceptronGanClient(train, test, seq_length)
    # p = multiprocessing.Process(target=run_parallel_compute, args=(gan, shared_results))
    # jobs.append(p)
    # p.start()


    # DAN
    # dan = DanClient(train, test, seq_length)
    # p = multiprocessing.Process(target=run_parallel_compute, args=(dan, shared_results))
    # jobs.append(p)
    # p.start()

    # Structured perceptron
    perceptron_client = PerceptronClient(train, test, seq_length)
    perceptron_client.add_constraints(constraints)
    p = multiprocessing.Process(target=run_parallel_compute, args=(perceptron_client, shared_results))
    jobs.append(p)
    p.start()

    # join all subprocesses
    for job in jobs:
        job.join()
    # return {'local': lc_acc, 'ffn': dl_acc, 'perceptron': p_acc, 'gan': gan_acc, 'tf': tf_acc, 'dan': dan_acc}
    # return shared threadlocal data
    print "RESULTS:"
    print shared_results
    return shared_results._getvalue()


# the main function
if __name__ == "__main__":
    results = run(10, 2000, num_constraints=10, soft=True)
    print "-------------------------------------------------------"
    print "RESULTS"
    print results
