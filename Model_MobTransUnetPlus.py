import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from keras.applications import MobileNetV3Small
from keras.models import Model
import numpy as np
from Evaluation import seg_evaluation


# Define the MobileNetV3-based TransUNet+ model
def create_transunet_model(input_shape):
    # Load the MobileNetV3Small model without the top classification layer
    base_model = MobileNetV3Small(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Encoder
    encoder_output = base_model.layers[-1].output

    # Decoder
    skip_connection_indices = [3, 6, 9]  # Indices of the MobileNetV3 blocks to use as skip connections

    x = encoder_output
    skip_connections = []
    for i in range(len(base_model.layers)):
        x = base_model.layers[i](x)
        if i in skip_connection_indices:
            skip_connections.append(x)

    x = UpSampling2D(size=(8, 8))(x)

    for skip_connection in reversed(skip_connections):
        x = Concatenate()([x, skip_connection])
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)

    # Output
    x = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def Model_MobTransUnetPlus(train_images, train_masks, test_images, test_masks, Images):
    # Specify the input shape
    input_shape = (224, 224, 3)  # Adjust according to the input size of the MobileNetV3Small model

    # Create the MobileNetV3-based TransUNet+ model
    model = create_transunet_model(input_shape)

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
    return segmented, Ev
