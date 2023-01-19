"""Constructs ResNet-50 neural network architecture."""
import tensorflow as tf
from utilities.architecture.residual_blocks import IdentityBlock, ConvolutionalBlock

def build_model():
    """ Build model with ResNet architecture.

    Returns:
        Tensorflow model.
    """
    # Input layer.
    input_layer = tf.keras.layers.Input(shape=(224,224,3), name='input_layer')

    # Zero padding.
    padding_layer = tf.keras.layers.ZeroPadding2D(padding=(3,3), name='padding_layer')(input_layer)

    # Input convolution.
    conv_input = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(7,7),
        strides=(2,2),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        name='first_convolutional_layer'
    )(padding_layer)
    conv_input_batchnorm = tf.keras.layers.BatchNormalization(axis=3, name='first_conv_batchnorm')(conv_input)
    conv_input_activation = tf.keras.layers.ReLU(name='first_conv_activation')(conv_input_batchnorm)
    conv_input_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        name='first_conv_maxpooling'
    )(conv_input_activation)

    # First set of residual blocks.
    conv_block_1 = ConvolutionalBlock(
        kernel_size=3,
        filters=[64, 64, 256],
        stride=1,
        block_name='conv_block_1'
    )(conv_input_maxpool)
    identity_block_1a = IdentityBlock(
        kernel_size=3,
        filters=[64, 64, 256],
        block_name='identity_block_1a'
    )(conv_block_1)
    identity_block_1b = IdentityBlock(
        kernel_size=3,
        filters=[64, 64, 256],
        block_name='identity_block_1b'
    )(identity_block_1a)

    # Second set of residual blocks.
    conv_block_2 = ConvolutionalBlock(
        kernel_size=3,
        filters=[128,128,512],
        stride=2,
        block_name='conv_block_2'
    )(identity_block_1b)
    identity_block_2a = IdentityBlock(
        kernel_size=3,
        filters=[128,128,512],
        block_name='identity_block_2a'
    )(conv_block_2)
    identity_block_2b = IdentityBlock(
        kernel_size=3,
        filters=[128,128,512],
        block_name='identity_block_2b'
    )(identity_block_2a)
    identity_block_2c = IdentityBlock(
        kernel_size=3,
        filters=[128,128,512],
        block_name='identity_block_2c'
    )(identity_block_2b)

    # Third set of residual blocks.
    conv_block_3 = ConvolutionalBlock(
        kernel_size=3,
        filters=[256,256,1024],
        stride=2,
        block_name='conv_block_3'
    )(identity_block_2c)
    identity_block_3a = IdentityBlock(
        kernel_size=3,
        filters=[256,256,1024],
        block_name='identity_block_3a'
    )(conv_block_3)
    identity_block_3b = IdentityBlock(
        kernel_size=3,
        filters=[256,256,1024],
        block_name='identity_block_3b'
    )(identity_block_3a)
    identity_block_3c = IdentityBlock(
        kernel_size=3,
        filters=[256,256,1024],
        block_name='identity_block_3c'
    )(identity_block_3b)
    identity_block_3d = IdentityBlock(
        kernel_size=3,
        filters=[256,256,1024],
        block_name='identity_block_3d'
    )(identity_block_3c)
    identity_block_3e = IdentityBlock(
        kernel_size=3,
        filters=[256,256,1024],
        block_name='identity_block_3e'
    )(identity_block_3d)

    # Fourth set of residual blocks.
    conv_block_4 = ConvolutionalBlock(
        kernel_size=3,
        filters=[512,512,2048],
        stride=2,
        block_name='conv_block_4'
    )(identity_block_3e)
    identity_block_4a = IdentityBlock(
        kernel_size=3,
        filters=[512,512,2048],
        block_name='identity_block_4a'
    )(conv_block_4)
    identity_block_4b = IdentityBlock(
        kernel_size=3,
        filters=[512,512,2048],
        block_name='identity_block_4b'
    )(identity_block_4a)

    # Average pooling.
    average_pooling = tf.keras.layers.AveragePooling2D(
        pool_size=(2,2),
        padding='same',
        name='average_pooling_layer'
    )(identity_block_4b)

    # Output layer.
    flatten_layer = tf.keras.layers.Flatten()(average_pooling)
    output_layer = tf.keras.layers.Dense(
        units=1000,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        activation='softmax',

        name='output_layer'
    )(flatten_layer)

    # Create model.
    model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])

    return model
