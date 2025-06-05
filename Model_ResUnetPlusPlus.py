import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
import numpy as np
from Evaluation import seg_evaluation


# Define the Residual Block
def residual_block(input_tensor, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same')(input_tensor)
    x = Conv2D(filters, 3, activation=None, padding='same')(x)
    x = tf.keras.layers.add([input_tensor, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Define the ResUNet++ model
def create_resunet_plusplus_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = residual_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = residual_block(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = residual_block(conv3, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = residual_block(conv4, 512)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = Conv2D(256, 2, activation='relu', padding='same')(up1)
    merge1 = Concatenate()([conv3, up1])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge1)
    conv5 = residual_block(conv5, 256)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = Conv2D(128, 2, activation='relu', padding='same')(up2)
    merge2 = Concatenate()([conv2, up2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv6 = residual_block(conv6, 128)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = Conv2D(64, 2, activation='relu', padding='same')(up3)
    merge3 = Concatenate()([conv1, up3])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv7 = residual_block(conv7, 64)

    # Output
    output = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=output)
    return model


def Model_ResUnetPlusPlus(train_images, train_masks, test_images, test_masks, Images):
    # Specify the input shape
    input_shape = (224, 224, 3)  # Adjust according to your input size

    # Create the ResUNet++ model
    model = create_resunet_plusplus_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare random training data and targets for demonstration
    train_masks = np.expand_dims(train_masks, axis=-1)

    # Prepare random training data and targets for demonstration
    test_masks = np.expand_dims(test_masks, axis=-1)

    # Train the model
    model.fit(train_images, train_masks, epochs=10, batch_size=16)  # Adjust the batch size and number of epochs as needed


    # Make predictions on test images
    predictions = model.predict(test_images)
    Ev = []
    for i in range(len(predictions)):
        Ev.append(seg_evaluation(predictions[i], test_masks[i]))
    segmented = model.predict(Images)
    return Ev
