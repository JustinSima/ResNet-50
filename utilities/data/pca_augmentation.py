""" Functions to create and apply PCA transformations."""
import os
import numpy as np
import tensorflow as tf

def create_pca_term(eig_values : np.array, eig_vectors : np.array):
    """Creates PCA term as described in section 4.1

    Args:
        eig_values (np.array: Array of eigenvalues.
        eig_vectors (np.array): Array of eigenvectors.

    Returns:
        np.array: Randomly jittered PCA term.
    """
    # Create random alpha coefficients.
    alphas = np.random.normal(loc=0.0, scale=0.1, size=3)

    # Create PCA term.
    pca_term = np.dot(eig_vectors, (eig_values * alphas))

    return pca_term

def calculate_pca_constants(directory : str) -> tuple:
    """ Returns eigenvalues and eigenvectors from PCA decompostion of RGB values for
    all provided image names.
    Used to find constants needed for preprocessing RGB values during training and fitting.

    Args:
        image_directory (str): Directory containing images.

    Returns:
        tuple: (eigenvalues : np.array, eigenvectors : np.array, rgb_means : np.array).

        3X1 array, 3x3 array, and 3x1 array containing the eigenvalues, eigenvectors, and means,
        respectively, for the centering and PCA decomposition of RGB values.
    """
    # Find all JPEG files in directory.
    image_names = []
    for _, _, image_list in os.walk(directory):
        for image_name in image_list:
            if image_name.endswith('.JPEG'):
                image_names.append(image_name)

    # Create array of all rgb values in directory.
    rgb_array = np.empty((3,1))
    for image_name in image_names:
        file_name = os.path.join(directory, image_name)
        image = tf.keras.preprocessing.image.load_img(file_name, target_size=[256, 256])

        image_array = tf.keras.preprocessing.image.img_to_array(image)
        rgb_values = image_array[2].reshape((3, -1))

        rgb_array = np.concatenate([rgb_array, rgb_values], axis=1)

    # Center and scale data.
    rgb_means = np.mean(rgb_array, axis=1)
    rgb_array[0] -= rgb_means[0]
    rgb_array[1] -= rgb_means[1]
    rgb_array[2] -= rgb_means[2]

    rgb_array = rgb_array / 255.

    # Find eigenvalues and eigevectors of covariance matrix.
    covariance_matrix = np.cov(rgb_array)
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

    return eig_values, eig_vectors, rgb_means
