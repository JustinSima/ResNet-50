"""Load and prepare images from ImageNet dataset."""
import os
import json
import numpy as np
import tensorflow as tf

from utilities.data.pca_augmentation import create_pca_term

class ImageNetDataGenerator(tf.keras.utils.Sequence):
    """ Tensorflow data generator.

    Attributes:
        directory : Directory containing images.
        list_ids : Unique names of images.
        labels : Path to json mappting image name (id) to corresponding label.
        label_encoding : Path to json mapping label to label index.
    """
    def __init__(
            self,
            image_directory, label_path, label_encoding_path,
            eigenvalues, eigenvectors, rgb_means,
            batch_size=2, shape=(224, 224), n_channels=3, n_classes=1000, shuffle=True
        ):
        # General attributes.
        self.shape = shape
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        # Image directory, image indexes (ids), labels, and encodings.
        self.directory = image_directory
        self.list_ids = [
            picture_name for picture_name in
            os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, picture_name))
        ]
        self.labels = load_json(label_path)
        self.label_encoding = load_json(label_encoding_path)

        # Terms for PCA augmentation.
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.rgb_means = rgb_means
        self.on_epoch_end()

    def __len__(self):
        """ Returns the number of batches per epoch."""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index : list):
        """ Generate a single batch."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of ids
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def __data_generation(self, list_ids_temp):
        """ Generates data containing batch_size samples.

        Args:
            list : List of image ids for batch.

        Returns:
            tuple : (
                X : (n_samples, *shape, n_channels),
                y : (n_samples, n_classes)
                )
        """
        # Initialization
        X = np.empty((self.batch_size, *self.shape, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, id_ in enumerate(list_ids_temp):
            # Store sample
            image_path = os.path.join(self.directory, id_)

            # Transform sample.
            median_term = np.zeros((224, 224, 3))
            median_term[2] = np.asarray(self.rgb_means)

            pca_term = np.zeros((224, 224, 3))
            pca_term[2] = create_pca_term(self.eigenvalues, self.eigenvectors)

            image_array = ( (process_image(image_path) - median_term) / 255. ) + pca_term

            ######
            X[i,] = image_array

            # Store class
            label_id = self.labels[id_[:9]]
            label_index = self.label_encoding[label_id]

            y_ = np.zeros(self.n_classes)
            y_[label_index] = 1
            y[i,] = y_

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

def process_image(file_name : str) -> np.array:
    """ Load and prepare image for model.
    Randomly selects 224x224 slice of image, with a 0.5 probability of being flipped horizontally.

    Args:
        file_name (str): Path to image.

    Returns:
        np.array: Numpy array representation of image, with size (224, 224, 3).
    """
    # Retrieve image from file name.
    image = tf.keras.preprocessing.image.load_img(file_name)
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Rescale image.
    rescaled_image, rescaled_dimensions = randomly_rescale_image(image_array, 256, 480)
    image_width = rescaled_dimensions[0]
    image_height = rescaled_dimensions[1]

    # Select cropped slice of image.
    corner = ( np.random.randint(0, image_width-224), np.random.randint(0, image_height-224) )
    image_cropped = rescaled_image[
        corner[0]:corner[0]+224,
        corner[1]:corner[1]+224,
        :
    ]

    # Flip image with probability 0.5.
    flip_image = np.random.choice([True, False])
    if flip_image:
        image_cropped = tf.image.flip_left_right(image_cropped)

    return image_cropped

def randomly_rescale_image(image, lower_bound, upper_bound):
    # Randomly choose scale.
    short_side_scale = np.random.randint(lower_bound, upper_bound + 1)

    # Find height and width, find the shortest side length.
    original_width = tf.shape(image)[0]
    original_height = tf.shape(image)[1]
    min_length = tf.minimum(original_height, original_width)

    # Find and apply scaling ratio.
    scaling_ratio = min_length / short_side_scale
    output_width = tf.cast(original_width, dtype=tf.float32) / tf.cast(scaling_ratio, dtype=tf.float32)
    output_height = tf.cast(original_height, dtype=tf.float32) / tf.cast(scaling_ratio, dtype=tf.float32)
    output_shape = [output_width, output_height]
    output_image = tf.image.resize(image, output_shape)

    return output_image, output_shape

def rescale_image(image, short_side_scale):
    # Find height and width, find the shortest side length.
    original_width = tf.shape(image)[0]
    original_height = tf.shape(image)[1]
    min_length = tf.minimum(original_height, original_width)

    # Find and apply scaling ratio.
    scaling_ratio = min_length / short_side_scale
    output_width = tf.cast(original_width, dtype=tf.float32) / tf.cast(scaling_ratio, dtype=tf.float32)
    output_height = tf.cast(original_height, dtype=tf.float32) / tf.cast(scaling_ratio, dtype=tf.float32)
    output_shape = [output_width, output_height]
    output_image = tf.image.resize(image, output_shape)

    return output_image, output_shape

def load_json(file_path) -> dict:
    """Load json as dictionary.

    Args:
        file_path (str): Path to json file.

    Returns:
        dict: Dictionary resulting from json file.
    """
    with open(file_path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    return data

def image_to_input(
    file_name : str,
    eig_values : np.array,
    eig_vectors : np.array,
    rgb_means: np.array
) -> np.array:
    """ Create ten 224x224 prediction images from single 256x256 input image.

    Args:
        file_name (str): Path to input image.

    Returns:
        np.array: Array to be passed to self.model predict method (batch_size, *shape, n_channels).
    """
    X_predict = np.empty((50, 224, 224, 3))
    mean_term = np.zeros((224, 224, 3))
    mean_term[3] = np.asarray(rgb_means)

    for rescale_number, short_side_length in enumerate([224, 256, 384, 480, 640]):
        # Retrieve image from file name.
        original_image = tf.keras.preprocessing.image.load_img(file_name)
        image_array = tf.keras.preprocessing.image.img_to_array(original_image)

        # Rescale image.
        rescaled_image, rescaled_dimensions = rescale_image(image_array, short_side_length)
        image_width = rescaled_dimensions[0]
        image_height = rescaled_dimensions[1]

        # Select cropped slices of image.
        corners = [
            (0, 0), (image_width-224, 0), # Top corners.
            (np.floor(image_width/2) - 112, np.floor(image_height/2) - 112), # Middle.
            (0, image_height-224), (image_width-224, image_height-224) # Bottom corners.
        ]
        for idx, corner in enumerate(corners):
            image_cropped = rescaled_image[
                int(corner[0]):int(corner[0]+224),
                int(corner[1]):int(corner[1]+224),
                :
            ]
            image_cropped_flipped = tf.image.flip_left_right(image_cropped)

            # Center, scale, and jitter RGB values.
            pca_term = np.zeros((224, 224, 3))
            pca_term[2] = create_pca_term(eig_values, eig_vectors)
            image_cropped = ( (image_cropped - mean_term) / 255. ) + pca_term
            image_cropped_flipped = ( (image_cropped_flipped - mean_term) / 255. ) + pca_term

            X_predict[10 * rescale_number + 2 * idx] = image_cropped
            X_predict[10 * rescale_number + 2 * idx + 1] = image_cropped_flipped

    return X_predict
