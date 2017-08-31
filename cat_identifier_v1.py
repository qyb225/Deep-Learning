import numpy as np
from PIL import Image
from core.learning_model import LR, NeuralNetwork
from core.image_model import image_processing

# 1. Data
def load_images_data():
    train_set_x = None
    test_set_x = None
    train_set_y = np.zeros((1, 60))
    test_set_y = np.zeros((1, 29))

    for i in range(30):
        print ('loading: cat_' + str(i) + '.jpg...')
        img = Image.open('./images/cat/cat_' + str(i) + '.jpg')
        if i == 0:
            train_set_x = image_processing.image_to_vector(img, 75, 100)
        else:
            train_set_x = np.column_stack((train_set_x, image_processing.image_to_vector(img, 75, 100)))
        train_set_y[0][i] = 1

    for i in range(30, 40):
        print ('loading: cat_' + str(i) + '.jpg...')
        img = Image.open('./images/cat/cat_' + str(i) + '.jpg')
        if i == 30:
            test_set_x = image_processing.image_to_vector(img, 75, 100)
        else:
            test_set_x = np.column_stack((test_set_x, image_processing.image_to_vector(img, 75, 100)))
        test_set_y[0][i - 30] = 1

    for i in range(30):
        print ('loading: other_' + str(i) + '.jpg...')
        img = Image.open('./images/other/' + str(i) + '.jpg')
        train_set_x = np.column_stack((train_set_x, image_processing.image_to_vector(img, 75, 100)))

    for i in range(30, 49):
        print ('loading: other_' + str(i) + '.jpg...')
        img = Image.open('./images/other/' + str(i) + '.jpg')
        test_set_x = np.column_stack((test_set_x, image_processing.image_to_vector(img, 75, 100)))

    return train_set_x, train_set_y, test_set_x, test_set_y

def run():
    train_set_x, train_set_y, test_set_x, test_set_y = load_images_data()

    # print ("LR Run...")
    # lr = LR.LR(train_set_x, train_set_y, lambda x : x / 255)
    # lr.train_model_run()
    # lr.predict_model_run(train_set_x, train_set_y)
    # lr.predict_model_run(test_set_x, test_set_y)

    # print ("NN1 Run...")
    # nn = NeuralNetwork.OneHiddenLayerNN(5, train_set_x, train_set_y, lambda x : x / 255)
    # nn.train_model_run(1501, 0.01)
    # nn.predict_model_run(train_set_x, train_set_y)
    # nn.predict_model_run(test_set_x, test_set_y)

    print ("NN2 Run...")
    nn2 = NeuralNetwork.NN(train_set_x, train_set_y, [25, 18, 15], lambda x : x / 255)
    nn2.train_model_run(1501, 0.011)
    nn2.predict_model_run(train_set_x, train_set_y)
    nn2.predict_model_run(test_set_x, test_set_y)

run()