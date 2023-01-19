""" Residual blocks."""
import tensorflow as tf

class IdentityBlock(tf.keras.Model):
    """ Create identity bottleneck block with given filters, and kernel size."""
    def __init__(self, kernel_size, filters, block_name):
        super(IdentityBlock, self).__init__(name=block_name)
        filter_1, filter_2, filter_3 = filters

        # Relu activation.
        self.activation = tf.keras.activations.relu

        # Define layers.
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=filter_1,
            kernel_size=(1, 1),
            strides=(1,1),
            padding='valid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_1'
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_1'
        )

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=filter_2,
            kernel_size=( kernel_size, kernel_size ),
            strides=(1,1),
            padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_2'
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_2'
        )

        self.conv_3 = tf.keras.layers.Conv2D(
            filters=filter_3,
            kernel_size=(1, 1),
            strides=(1,1),
            padding='valid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_3')
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_3'
        )

    def call(self, input_tensor, training=False):
        """ Perform forward pass."""
        x_skip = input_tensor

        x_intermediate = self.conv_1(input_tensor)
        x_intermediate = self.batch_norm_1(x_intermediate, training=training)
        x_intermediate = self.activation(x_intermediate)

        x_intermediate = self.conv_2(x_intermediate)
        x_intermediate = self.batch_norm_2(x_intermediate, training=training)
        x_intermediate = self.activation(x_intermediate)

        x_intermediate = self.conv_3(x_intermediate)
        x_intermediate = self.batch_norm_3(x_intermediate, training=training)

        x_intermediate += x_skip

        x_output = self.activation(x_intermediate)

        return x_output

class ConvolutionalBlock(tf.keras.Model):
    """ Create convolutional block with given filters, kernel size, and stride length."""
    def __init__(self, kernel_size, filters, stride, block_name):
        super(ConvolutionalBlock, self).__init__(name=block_name)
        filter_1, filter_2, filter_3 = filters

        # Activation.
        self.activation = tf.keras.activations.relu

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=filter_1,
            kernel_size=(1,1),
            strides=(stride, stride),
            padding='valid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_1'
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_1'
        )

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=filter_2,
            kernel_size=(kernel_size, kernel_size),
            strides=(1,1),
            padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_2'
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_2'
        )

        self.conv_3 = tf.keras.layers.Conv2D(
            filters=filter_3,
            kernel_size=(1,1),
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_3'
        )
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_3'
        )

        self.conv_skip = tf.keras.layers.Conv2D(
            filters=filter_3,
            kernel_size=(1,1),
            strides=(stride, stride),
            padding='valid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
            name=block_name+'_conv_skip'
        )
        self.batch_norm_skip = tf.keras.layers.BatchNormalization(
            axis=3,
            name=block_name+'_batch_norm_skip'
        )

    def call(self, input_tensor, training=False):
        """ Perform forward pass."""
        x_skip = input_tensor

        x_intermediate = self.conv_1(input_tensor)
        x_intermediate = self.batch_norm_1(x_intermediate, training=training)
        x_intermediate = self.activation(x_intermediate)

        x_intermediate = self.conv_2(x_intermediate)
        x_intermediate = self.batch_norm_2(x_intermediate, training=training)
        x_intermediate = self.activation(x_intermediate)

        x_intermediate = self.conv_3(x_intermediate)
        x_intermediate = self.batch_norm_3(x_intermediate, training=training)

        x_skip = self.conv_skip(x_skip)
        x_skip = self.batch_norm_skip(x_skip)

        x_intermediate += x_skip

        x_output = self.activation(x_intermediate)

        return x_output
