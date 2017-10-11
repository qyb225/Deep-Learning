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
    def __init__(self, train_set_x, train_set_y, layer_dims, standardize, activation_fn = "relu"):
        """
        :param train_set_x: training set X (n, m)
        :param train_set_y: training set Y (1, m)
        :param layer_dims:  python array (list) containing the dimensions of each layer in our network
        :param standardize: standardize function
        :param activation_fn: activation function
        """
        self._standardize = standardize
        self._train_set_x = self._standardize_data(train_set_x)
        self._train_set_y = train_set_y
        self._activation_fn = activation_fn
        self._params = self._init_params(layer_dims)

    def _standardize_data(self, X):
        return self._standardize(X)

    def _init_params(self, layer_dims):
        params = []
        n_x = self._train_set_x.shape[0]
        n_y = self._train_set_y.shape[0]
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
    def _cost_function(self, y_hat, lambd):
        m = self._train_set_x.shape[1]

        logprobs = np.multiply(np.log(y_hat), self._train_set_y) \
                   + np.multiply(np.log(1 - y_hat), 1 - self._train_set_y)

        cost = -1 / m * np.sum(logprobs)

        #Regularzation
        reg = 0
        for i in range(len(self._params)):
            reg += np.sum(np.multiply(self._params[i]['W'], self._params[i]['W']))
        cost += lambd / (2 * m) * reg

        return cost

    #training
    def _liner_forward(self, A, W, b):
        Z = np.dot(W, A) + b

        return Z

    def _linear_activation_forward(self, A_prev, W, b, activation, keep_prop):
        Z = self._liner_forward(A_prev, W, b)
        A = None
        if activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "relu":
            A = relu(Z)
        elif activation == "tanh":
            A = tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prop
        A = np.multiply(A, D)
        A /= keep_prop
        return A, Z, D

    def _forward_propagation(self, X, keep_prop = 1):
        A = X
        caches = []
        L = len(self._params)
        for i in range(L - 1):
            A, Z, D = self._linear_activation_forward(A, self._params[i]['W'], self._params[i]['b'], self._activation_fn, keep_prop)
            caches.append({
                'A': A,
                'Z': Z,
                'D': D
            })

        #Final layer keep_prop must be 1.
        AL, ZL, DL = self._linear_activation_forward(A, self._params[L - 1]['W'], self._params[L - 1]['b'], "sigmoid", 1)
        caches.append({
            'A': AL,
            'Z': ZL,
            'D': DL
        })
        return  AL, caches

    def _liner_backward(self, dZ, A_prev, W, b):
        m = self._train_set_y.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, A_prev, Z, D, W, b, activation, keep_prop, lambd):
        dA = np.multiply(dA, D)
        dA /= keep_prop
        dZ = None
        if activation == "relu":
            dZ = dA * relu_prime(Z)
        elif activation == "tanh":
            dZ = dA * tanh_prime(Z)
        elif activation == "sigmoid":
            dZ = dA * sigmoid_prime(Z)

        dA_prev, dW, db =  self._liner_backward(dZ, A_prev, W, b)
        m = self._train_set_x.shape[1]
        dW += lambd / m * W

        return dA_prev, dW, db

    def _backward_propagation(self, AL, caches, keep_prop, lambd):
        grads = []
        dAL = -1 * self._train_set_y / AL + (1 - self._train_set_y) / (1 - AL)
        L = len(caches)

        Z = caches[L - 1]['Z']
        D = caches[L - 1]['D']
        A_prev = caches[L - 2]['A']
        W = self._params[L - 1]['W']
        b = self._params[L - 1]['b']
        dA_prev, dW, db = self._linear_activation_backward(dAL, A_prev, Z, D, W, b, "sigmoid", 1, lambd)
        grads.insert(0, {
            'dW': dW,
            'db': db
        })

        for i in reversed(range(1, L - 1)):
            Z = caches[i]['Z']
            D = caches[i]['D']
            A_prev = caches[i - 1]['A']
            W = self._params[i]['W']
            b = self._params[i]['b']
            dA_prev, dW, db = self._linear_activation_backward(dA_prev, A_prev, Z, D, W, b, self._activation_fn, keep_prop, lambd)
            grads.insert(0, {
                'dW': dW,
                'db': db
            })

        Z = caches[0]['Z']
        D = caches[0]['D']
        A_prev = self._train_set_x
        W = self._params[0]['W']
        b = self._params[0]['b']
        dA_prev, dW, db = self._linear_activation_backward(dA_prev, A_prev, Z, D, W, b, self._activation_fn, keep_prop, lambd)
        grads.insert(0, {
            'dW': dW,
            'db': db
        })

        return grads

    def _update_params(self, grads, learning_rate):
        L = len(grads)

        for i in range(L):
            self._params[i]['W'] -= learning_rate * grads[i]['dW']
            self._params[i]['b'] -= learning_rate * grads[i]['db']
        return

    def _training_iterate(self, num_iterations, learning_rate, keep_prop, lambd):
        for i in range(0, num_iterations):
            Y_hat, caches = self._forward_propagation(self._train_set_x, keep_prop)
            cost = self._cost_function(Y_hat, lambd)
            grads = self._backward_propagation(Y_hat, caches, keep_prop, lambd)
            self._update_params(grads, learning_rate)

            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return cost

    def _predict(self, X):
        y_hat, caches = self._forward_propagation(X)
        Y_prediction = np.array(y_hat > 0.5, int)
        return Y_prediction

    # API
    def train_model_run(self, num_iterations = 1001, learning_rate = 0.01, keep_prop = 0.8, lambd = 0.7):
        self._training_iterate(num_iterations, learning_rate, keep_prop, lambd)

    def predict_model_run(self, data_x, data_y):
        X = self._standardize_data(data_x)
        Y_prediction = self._predict(X)
        print ("Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - data_y)) * 100))

        return Y_prediction
