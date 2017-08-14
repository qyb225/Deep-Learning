import numpy as np
from PIL import Image

# 1. Data
def image_to_matrix(image, width = 0, height = 0):
    if width == 0 or height == 0:
        width = image.size[0]
        height = image.size[1]
    img = image.resize((width, height))
    mat = np.zeros((width * height, 3))

    for i in range(width):
        for j in range(height):
            pix = img.getpixel((i, j))
            mat[i * height + j] = pix
    return mat

def image_to_vector(image, width = 0, height = 0):
    mat = image_to_matrix(image, width, height)
    return mat.reshape((mat.shape[0] * mat.shape[1], 1))


def load_images_data():
    train_set_x = None
    test_set_x = None
    train_set_y = np.zeros((1, 60))
    test_set_y = np.zeros((1, 29))

    for i in range(30):
        print ('loading: cat_' + str(i) + '.jpg...')
        img = Image.open('./images/cat/cat_' + str(i) + '.jpg')
        if i == 0:
            train_set_x = image_to_vector(img, 75, 100)
        else:
            train_set_x = np.column_stack((train_set_x, image_to_vector(img, 75, 100)))
        train_set_y[0][i] = 1

    for i in range(30, 40):
        print ('loading: cat_' + str(i) + '.jpg...')
        img = Image.open('./images/cat/cat_' + str(i) + '.jpg')
        if i == 30:
            test_set_x = image_to_vector(img, 75, 100)
        else:
            test_set_x = np.column_stack((test_set_x, image_to_vector(img, 75, 100)))
        test_set_y[0][i - 30] = 1

    for i in range(30):
        print ('loading: other_' + str(i) + '.jpg...')
        img = Image.open('./images/other/' + str(i) + '.jpg')
        train_set_x = np.column_stack((train_set_x, image_to_vector(img, 75, 100)))

    for i in range(30, 49):
        print ('loading: other_' + str(i) + '.jpg...')
        img = Image.open('./images/other/' + str(i) + '.jpg')
        test_set_x = np.column_stack((test_set_x, image_to_vector(img, 75, 100)))

    return train_set_x, train_set_y, test_set_x, test_set_y

# Train

def sigmoid(z):
    ans = 1 / (1 + np.exp(-1 * z))
    return ans

def init_params(dim_w):
    b = 0
    w = np.zeros((dim_w, 1))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    y_hat = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * (Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat)).sum()

    dw = 1 / m * np.dot(X, (y_hat - Y).T)
    db = 1 / m * (y_hat - Y).sum()

    grads = {'dw': dw, 'db': db}

    return grads, cost


def training_iterate(w, b, X, Y, num_iterations = 1001, learning_rate = 0.01):
    cost = None
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            print ("Cost after iteration %i: %lf" %(i, cost))

    params = { 'w': w, 'b': b }
    return params, cost

def predict(w, b, X):
    Y_prediction = np.zeros((1, X.shape[1]))
    y_hat = sigmoid(np.dot(w.T, X) + b)

    for i in range(y_hat.shape[1]):
        if y_hat[0][i] <= 0.5:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1

    return Y_prediction

def model():
    train_set_x, train_set_y, test_set_x, test_set_y = load_images_data()

    #Let's standardize our dataset.
    train_set_x = train_set_x / 255
    test_set_x = test_set_x / 255

    w, b = init_params(train_set_x.shape[0])
    params, cost = training_iterate(w, b, train_set_x, train_set_y)
    w = params['w']
    b = params['b']

    y_pred_train = predict(w, b, train_set_x)
    print ("Train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - train_set_y)) * 100))

    y_pred_test = predict(w, b, test_set_x)
    print ("Test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - test_set_y)) * 100))


model()