# Code to make tensorflow-gpu run with RTX 2080
# --------------------------------------------------
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# --------------------------------------------------

import numpy as np
import tensorflow as tf
from os.path import join
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from Helpers import helpers as h


def train_rhydon_detect(_checkpoint_name: str, _checkpoint_dir: str='RhydonMate/Models/training/ckpt/ckpt',
                        _load_model: bool=False, _load_model_dir: str='', _test_pct: float = 0.2) -> float:
    print(not _load_model or _load_model_dir != '')
    assert not _load_model or _load_model_dir != '', 'Enter a valid directory for model loading.'

    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session() # Destroy TF graph and create a new one

    x_train, y_train, x_test, y_test, x_valid, y_valid = h.get_image_data(
        detect_name   = 'Rhydon',
        image_dir     = 'RhydonMate/Data/UE4',
        data_dir      = 'RhydonMate/Data/ImageData.json',
        test_pct      = _test_pct,
        valid_pct     = 0.0,
        use_grayscale = True,
        dataset       = 'pokemonDataSet',
        normalize     = True
    )

    # Shape data (images are 270 x 480 px)
    x_train = x_train.reshape(-1, 270, 480, 1)
    y_train = y_train.reshape(-1, 1)
    x_test  = x_test .reshape(-1, 270, 480, 1)
    y_test  = y_test .reshape(-1, 1)
    x_valid = x_valid.reshape(-1, 270, 480, 1)
    y_valid = y_valid.reshape(-1, 1)

    # Create a callback that saves checkpoints during training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=join(_checkpoint_dir + "_" + datetime.now().strftime("%Y%m%d-%M%S"), _checkpoint_name + '.ckpt'),
                                                 save_weights_only=True,
                                                 verbose=1)

    # Setup model
    model = load_model()

    # Must have model with exact same architecture when loading weights
    if (_load_model):
        model.load_weights(_load_model_dir)

    # Train model
    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'mse']
    )

    model.fit(x_train, y_train, epochs=5, callbacks=[cp_callback])

    # Evaluate model
    test_results = model.evaluate(x_test, y_test, verbose=1)
    # del x_train, y_train, x_test, y_test, x_valid, y_valid # Clear data from RAM

    return test_results[1]


def is_this_rhydon(_load_model_dir: str, _image_dir: str, _use_greyscale: bool=False, _normalize: bool=True):
    # Load model architecture
    model = load_model()

    # Load model weights (from past training)
    model.load_weights(_load_model_dir)

    # Perform inference
    prediction = model.predict(
        h.get_single_image(
            image_dir=_image_dir,
            use_grayscale=_use_greyscale,
            normalize=_normalize),
        )

    return prediction[0]


def load_model():
    """
    Returns CNN model architecture which is used for training and inference.
    """
    return Sequential([

        # 2D convolutional layer
        Conv2D(
            input_shape=(270, 480, 1),
            filters=64,
            kernel_size=[2, 2],
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        MaxPooling2D(),
        Conv2D(
            filters=32,
            kernel_size=[2, 2],
            strides=1,
            padding='same',
            activation='relu',
        ),
        MaxPooling2D(),
        Conv2D(
            filters=56,
            kernel_size=[2, 2],
            strides=1,
            padding='same',
            activation='relu',
        ),
        Flatten(),
        Dense(2, activation='softmax')
    ])
