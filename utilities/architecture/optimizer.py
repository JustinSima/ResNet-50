""" Model compiler."""
import tensorflow as tf
import tensorflow_addons as tfa

def compile_model(model):
    """ Compile model.
    Cost Function: Categorical cross entropy.
    Optimization Method: Stochastic gradient descent with momentum and constant factor weight decay.
    """
    # Stochastic gradient descent with momentum and weight decay.
    optimizer = tfa.optimizers.SGDW(
        weight_decay=0.0001,
        momentum=0.9,
        learning_rate=0.1
    )

    cost_function = tf.keras.losses.CategoricalCrossentropy()
    metric = 'categorical_crossentropy'

    model.compile(
        optimizer=optimizer,
        loss=cost_function,
        metrics=[metric]
    )
