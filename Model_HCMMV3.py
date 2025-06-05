import numpy as np
from Evaluation import evaluation
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.applications import MobileNetV3Small
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder


def Model_HCMMV3(train_images, train_targets, test_images, test_targets):
    image_size = 128
    num_classes = train_targets.shape[1]

    # Convert target labels to one-hot encoded format
    encoder = OneHotEncoder(sparse=False)
    train_targets_encoded = encoder.fit_transform(train_targets.reshape(-1, 1))
    test_targets_encoded = encoder.fit_transform(test_targets.reshape(-1, 1))

    # Define the Hybrid Convolution-based Multiscale MobileNetV3 model
    def create_multiscale_mobilenetv3_model(input_shape, num_classes):
        input_tensor = Input(shape=input_shape)

        # Multiscale Convolutions
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        x = tf.concat([conv1, conv2], axis=-1)  # Concatenate multiscale feature maps

        # MobileNetV3Small as feature extractor
        base_model = MobileNetV3Small(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            alpha=1.0,
            weights=None
        )
        base_output = base_model(x)

        # Global Average Pooling and Dense layers for classification
        x = GlobalAveragePooling2D()(base_output)
        x = Dense(128, activation='relu')(x)
        output_tensor = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        return model

    # Create the model
    input_shape = (image_size, image_size, 3)
    model = create_multiscale_mobilenetv3_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_targets_encoded, epochs=10, batch_size=32,
              validation_data=(test_images, test_targets_encoded))

    predict = model.predict(test_images)
    # Convert target labels to one-hot encoded format
    encoder = OneHotEncoder(sparse=False)
    predict = encoder.fit_transform(predict.reshape(-1, 1))
    Eval = evaluation(predict, test_targets_encoded)
    return Eval
