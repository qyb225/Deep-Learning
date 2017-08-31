import numpy as np

# Function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.array(z > 0, int)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)

# Model
class NN(object):
    #init with data set and layer dims
    def __init__(self, train_set_x, train_set_y, layer_dims, standardize):
        """
        :param train_set_x: training set X (n, m)
        :param train_set_y: training set Y (1, m)
        :param layer_dims:  python array (list) containing the dimensions of each layer in our network
        :param standardize: standardize function
        """
        self.standardize = standardize
        self.train_set_x = self.standardize_data(train_set_x)
        self.train_set_y = train_set_y
        self.params = self.init_params(layer_dims)

    def standardize_data(self, X):
        return self.standardize(X)

    def init_params(self, layer_dims):
        params = []
        n_x = self.train_set_x.shape[0]
        n_y = self.train_set_y.shape[0]
        params.append({
            'W': np.random.randn(layer_dims[0], n_x) * 0.01,
            'b': np.zeros((layer_dims[0], 1))
        })
        for i in range(1, len(layer_dims)):
            layer_params = {
                'W': np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01,
                'b': np.zeros((layer_dims[i], 1))
            }
            params.append(layer_params)
        params.append({
            'W': np.random.randn(n_y, layer_dims[len(layer_dims) - 1]),
            'b': np.zeros((n_y, 1))
        })
        return params

    # function
    def cost_function(self, y_hat):
        m = self.train_set_x.shape[1]

        logprobs = np.multiply(np.log(y_hat), self.train_set_y) \
                   + np.multiply(np.log(1 - y_hat), 1 - self.train_set_y)

        cost = -1 / m * np.sum(logprobs)
        return cost

    #training
    def liner_forward(self, A, W, b):
        Z = np.dot(W, A) + b

        return Z

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z = self.liner_forward(A_prev, W, b)
        A = None
        if activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "relu":
            A = relu(Z)
        elif activation == "tanh":
            A = tanh(Z)
        return A, Z

    def forward_propagation(self, X):
        A = X
        caches = []
        L = len(self.params)
        for i in range(L - 1):
            A, Z = self.linear_activation_forward(A, self.params[i]['W'], self.params[i]['b'], "relu")
            caches.append({
                'A': A,
                'Z': Z
            })

        AL, Z = self.linear_activation_forward(A, self.params[L - 1]['W'], self.params[L - 1]['b'], "sigmoid")
        caches.append({
            'A': AL,
            'Z': Z
        })
        return  AL, caches

    def liner_backward(self, dZ, A_prev, W, b):
        m = self.train_set_y.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, A_prev, Z, W, b, activation):
        dZ = None
        if activation == "relu":
            dZ = dA * relu_prime(Z)
        elif activation == "tanh":
            dZ = dA * tanh_prime(Z)
        elif activation == "sigmoid":
            dZ = dA * sigmoid_prime(Z)

        dA_prev, dW, db =  self.liner_backward(dZ, A_prev, W, b)
        return dA_prev, dW, db

    def backward_propagation(self, AL, caches):
        grads = []
        dAL = -1 * self.train_set_y / AL + (1 - self.train_set_y) / (1 - AL)
        L = len(caches)

        Z = caches[L - 1]['Z']
        A_prev = caches[L - 2]['A']
        W = self.params[L - 1]['W']
        b = self.params[L - 1]['b']
        dA_prev, dW, db = self.linear_activation_backward(dAL, A_prev, Z, W, b, "sigmoid")
        grads.insert(0, {
            'dW': dW,
            'db': db
        })

        for i in reversed(range(1, L - 1)):
            Z = caches[i]['Z']
            A_prev = caches[i - 1]['A']
            W = self.params[i]['W']
            b = self.params[i]['b']
            dA_prev, dW, db = self.linear_activation_backward(dA_prev, A_prev, Z, W, b, "relu")
            grads.insert(0, {
                'dW': dW,
                'db': db
            })

        Z = caches[0]['Z']
        A_prev = self.train_set_x
        W = self.params[0]['W']
        b = self.params[0]['b']
        dA_prev, dW, db = self.linear_activation_backward(dA_prev, A_prev, Z, W, b, "relu")
        grads.insert(0, {
            'dW': dW,
            'db': db
        })

        return grads

    def update_params(self, grads, learning_rate):
        L = len(grads)

        for i in range(L):
            self.params[i]['W'] -= learning_rate * grads[i]['dW']
            self.params[i]['b'] -= learning_rate * grads[i]['db']
        return

    def training_iterate(self, num_iterations, learning_rate):
        for i in range(0, num_iterations):
            Y_hat, caches = self.forward_propagation(self.train_set_x)
            cost = self.cost_function(Y_hat)
            grads = self.backward_propagation(Y_hat, caches)
            self.update_params(grads, learning_rate)

            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return cost

    def predict(self, X):
        y_hat, caches = self.forward_propagation(X)
        Y_prediction = np.array(y_hat > 0.5, int)
        return Y_prediction

    # API
    def train_model_run(self, num_iterations = 1001, learning_rate = 0.01):
        self.training_iterate(num_iterations, learning_rate)

    def predict_model_run(self, data_x, data_y):
        X = self.standardize_data(data_x)
        Y_prediction = self.predict(X)
        print ("Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - data_y)) * 100))

        return Y_prediction


# One Hidden Layer Neural Netowrk
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
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

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