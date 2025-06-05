import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from Evaluation import evaluation



def Model_Resnet(train_data, train_labels, test_data, test_labels):
    IMG_SIZE = 256
    train_images = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for n in range(train_data.shape[0]):
        train_images[n, :, :, :] = np.resize(train_data[n, :], [IMG_SIZE, IMG_SIZE, 1])

    test_images = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for n in range(test_data.shape[0]):
        test_images[n, :, :, :] = np.resize(test_data[n, :], [IMG_SIZE, IMG_SIZE, 1])
    # Generate random training data
    input_shape = (256, 256, 1)
    num_classes = int(len(np.unique(train_labels)))
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['mse'])

    # Train the model
    epochs = 10
    batch_size = 32

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

    # Perform prediction using the trained model
    predictions = model.predict(test_images)
    predicted_labels = int(np.argmax(predictions, axis=1).ravel())

    Eval = evaluation(predicted_labels, test_labels.ravel())
    return Eval