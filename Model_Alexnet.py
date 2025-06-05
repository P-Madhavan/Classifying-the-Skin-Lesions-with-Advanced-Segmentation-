import numpy as np
from Evaluation import evaluation
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

def Model_Alexnet(train_data, train_target, test_data, test_target):

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3),
                     kernel_size=(11, 11), strides=(4, 4),
                     padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flattening
    model.add(Flatten())

    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Softmax Layer
    model.add(Dense(train_target.shape[1]))
    model.add(Activation('softmax'))

    train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

    val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

    test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

    # Fitting the augmentation defined above to the data
    # Defining the parameters
    batch_size = 2
    epochs = 1
    learn_rate = .001

    IMG_SIZE = [224, 224, 3]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Feat1 = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Feat2 = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])


    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    # Training the model
    model.fit_generator(train_generator.flow(Feat1, train_target, batch_size=batch_size), epochs=epochs,
                          steps_per_epoch=1,
                          validation_data=val_generator.flow(Feat2, test_target, batch_size=batch_size), validation_steps=2,
                           verbose=1)
    # Making prediction
    y_pred = model.predict(Feat2)
    Eval = evaluation(y_pred, test_target)
    return Eval