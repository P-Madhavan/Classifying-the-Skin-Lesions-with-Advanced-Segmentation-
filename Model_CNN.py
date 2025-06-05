import numpy as np
from tensorflow.python.framework import ops
from keras.layers import Conv2D as conv_2d, MaxPooling2D as max_pool_2d
from keras.layers import Input as input_data, Dropout as dropout
from keras.layers import Flatten as fully_connected
from keras.estimator import model_to_estimator as regression
from Evaluation import evaluation


def Model_CNN(train_data, Y, test_data, test_y):
    IMG_SIZE = 20
    X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for n in range(train_data.shape[0]):
        X[n, :, :, :] = np.resize(train_data[n, :], [IMG_SIZE, IMG_SIZE, 1])

    test_x = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for n in range(test_data.shape[0]):
        test_x[n, :, :, :] = np.resize(test_data[n, :], [IMG_SIZE, IMG_SIZE, 1])

    LR = 1e-3
    ops.reset_default_graph()
    convnet = input_data(shape=[None, 20, 20, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 80, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=0.01,
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=5,
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    pred = model.predict(test_x)
    predict = np.zeros((test_y.shape[0], test_y.shape[1]))
    for i in range(test_y.shape[1]):
        out = np.round(pred[:,i])
        predict[:, i] = out
    eval = evaluation(test_y, predict)
    return eval
