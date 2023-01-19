"""Callbacks required for training."""
import tensorflow as tf

# Divide learning rate by ten upon plateau.
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,

    patience=10,
    mode='auto',
    min_delta=0.000001,
    min_lr=0
)

# Callbacks for model training.
CALLBACKS = [learning_rate_callback]
