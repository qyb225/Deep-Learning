__author__='Qubic'
import numpy as np

class LR(object):
    # init
    def __init__(self, train_set_x, train_set_y, standardize):
        self.standardize = standardize
        self.train_set_x = self.standardize_data(train_set_x)
        self.train_set_y = train_set_y
        w, b = self.init_params(train_set_x.shape[0])
        self.params = {'w': w, 'b': b}

    def standardize_data(self, X):
        return self.standardize(X)

    def sigmoid(self, z):
        ans = 1 / (1 + np.exp(-1 * z))
        return ans

    def init_params(self, dim_w):
        b = 0
        w = np.zeros((dim_w, 1))

        return w, b

    # training
    def propagate(self):
        m = self.train_set_x.shape[1]
        w = self.params['w']
        b = self.params['b']

        y_hat = self.sigmoid(np.dot(w.T, self.train_set_x) + b)
        cost = -1 / m * (self.train_set_y * np.log(y_hat) + (1 - self.train_set_y) * np.log(1 - y_hat)).sum()

        dw = 1 / m * np.dot(self.train_set_x, (y_hat - self.train_set_y).T)
        db = 1 / m * (y_hat - self.train_set_y).sum()

        grads = {'dw': dw, 'db': db}

        return grads, cost

    def training_iterate(self, num_iterations, learning_rate):
        cost = None
        for i in range(num_iterations):
            grads, cost = self.propagate()
            dw = grads['dw']
            db = grads['db']

            self.params['w'] = self.params['w'] - learning_rate * dw
            self.params['b'] = self.params['b'] - learning_rate * db

            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return cost

    # predict
    def predict(self, X):
        w = self.params['w']
        b = self.params['b']

        Y_prediction = np.zeros((1, X.shape[1]))
        y_hat = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(y_hat.shape[1]):
            if y_hat[0][i] <= 0.5:
                Y_prediction[0][i] = 0
            else:
                Y_prediction[0][i] = 1

        return Y_prediction

    # API
    def train_model_run(self, num_iterations = 1001, learning_rate = 0.01):
        self.training_iterate(num_iterations, learning_rate)

    def predict_model_run(self, data_x, data_y):
        X = self.standardize_data(data_x)
        Y_prediction = self.predict(X)
        print ("Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - data_y)) * 100))

        return Y_prediction