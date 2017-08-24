import numpy as np

class OneHiddenLayerNN(object):
    #init
    def __init__(self, n_hidden, train_set_x, train_set_y, standardize):
        self.standardize = standardize
        self.train_set_x = self.standardize_data(train_set_x)
        self.train_set_y = train_set_y
        n_x, n_h, n_y = self.layer_sizes(n_hidden)
        self.params = self.init_params(n_x, n_h, n_y)

    def standardize_data(self, X):
        return self.standardize(X)

    def layer_sizes(self, n_hidden):
        n_x = self.train_set_x.shape[0]
        n_h = n_hidden
        n_y = self.train_set_y.shape[0]

        return n_x, n_h, n_y

    def init_params(self, n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        params = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return params

    # function
    def sigmoid(self, z):
        ans = 1 / (1 + np.exp(-1 * z))
        return ans

    def cost_function(self, y_hat):
        m = self.train_set_x.shape[1]

        logprobs = np.multiply(np.log(y_hat), self.train_set_y) \
                   + np.multiply(np.log(1 - y_hat), 1 - self.train_set_y)

        cost = -1 / m * np.sum(logprobs)
        return cost

    #training
    def forward_propagation(self, X):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return cache

    def backward_propagation(self, cache):
        m = self.train_set_x.shape[1]
        A1 = cache['A1']
        A2 = cache['A2']

        W1 = self.params['W1']
        W2 = self.params['W2']

        dZ2 = A2 - self.train_set_y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1 / m * np.dot(dZ1, self.train_set_x.T)
        db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

        grads = {'dW1': dW1,
                 'db1': db1,
                 'dW2': dW2,
                 'db2': db2}
        return grads

    def update_params(self, grads, learning_rate):
        self.params['W1'] -= learning_rate * grads['dW1']
        self.params['b1'] -= learning_rate * grads['db1']
        self.params['W2'] -= learning_rate * grads['dW2']
        self.params['b2'] -= learning_rate * grads['db2']

    def training_iterate(self, num_iterations, learning_rate):
        for i in range(0, num_iterations):
            cache = self.forward_propagation(self.train_set_x)
            cost = self.cost_function(cache['A2'])
            grads = self.backward_propagation(cache)

            self.update_params(grads, learning_rate)

            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return cost

    def predict(self, X):
        y_hat = self.forward_propagation(X)['A2']
        Y_prediction = y_hat > 0.5
        return Y_prediction

    # API
    def train_model_run(self, num_iterations = 1001, learning_rate = 0.01):
        self.training_iterate(num_iterations, learning_rate)

    def predict_model_run(self, data_x, data_y):
        X = self.standardize_data(data_x)
        Y_prediction = self.predict(X)
        print ("Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - data_y)) * 100))

        return Y_prediction