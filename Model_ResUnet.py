import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
import numpy as np
from Evaluation import seg_evaluation

# Define the Residual Block
def residual_block(input_tensor, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.add([input_tensor, x])
    return x

# Define the ResUNet model
def create_resunet_model(input_shape):
    # Encoder
    inputs = tf.keras.Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = residual_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = residual_block(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = residual_block(conv3, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = residual_block(conv4, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = residual_block(conv5, 1024)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = residual_block(conv6, 512)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = residual_block(conv7, 256)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = residual_block(conv8, 128)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = residual_block(conv9, 64)

    # Output
    output = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=output)
    return model


def Model_ResUnet(train_images, train_masks, test_images, test_masks, Images):
    # Specify the input shape
    input_shape = (224, 224, 3)  # Adjust according to your input size

    # Create the ResUNet model
    model = create_resunet_model(input_shape)

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