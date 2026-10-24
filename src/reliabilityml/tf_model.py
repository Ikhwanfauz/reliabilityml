import numpy as np
import tensorflow as tf


def make_lstm_model(training_windows: np.ndarray) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    window_size = training_windows.shape[1]
    feature_count = training_windows.shape[2]

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(training_windows.reshape(-1, feature_count))

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(window_size, feature_count)),
            normalizer,
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
    )
    return model