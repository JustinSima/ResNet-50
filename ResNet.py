"""ResNet-50 model class."""
import os
import numpy as np

from utilities.data.data_generator import ImageNetDataGenerator, image_to_input
from utilities.architecture.model_builder import build_model
from utilities.architecture.optimizer import compile_model
from utilities.architecture.callbacks import CALLBACKS
from utilities.data.pca_augmentation import calculate_pca_constants

class ResNet:
    """Creates and trains model with ResNet-50 model
    architecture, data augmentations, and training/predictions schemes.

    Attributes:
        directory : Directory containing images.
        list_ids : Unique names of images.
        labels : Path to json mappting image name (id) to corresponding label.
        label_encoding : Path to json mapping label to label index.
        history : Model training history. Defined after calling fit.

    Methods:
        fit : Trains model for desired number of epochs.
        predict : Predicts label class distribution using ResNet procedure.
        predict_n : Predicts the n most likely class labels.
    """
    def __init__(self, directory, label_path, label_encoding_path):
        # Initialize model and callbacks.
        self.model = build_model()
        self.callbacks = CALLBACKS

        # Image directories.
        self.directory = directory
        self.train_directory = os.path.join(self.directory, 'train')
        self.val_directory = os.path.join(self.directory, 'val')
        self.test_directory = os.path.join(self.directory, 'test')

        # Paths to labels and encodings.
        self.label_path = label_path
        self.label_encoding_path = label_encoding_path

        # Create terms for PCA augmentations.
        self.eig_values, self.eig_vectors, self.rgb_means = calculate_pca_constants(
            self.train_directory
        )

        # Create data generators.
        self.train_generator = ImageNetDataGenerator(
            image_directory=self.train_directory,
            label_path=self.label_path,
            label_encoding_path=self.label_encoding_path,
            eigenvalues=self.eig_values,
            eigenvectors=self.eig_vectors,
            rgb_means=self.rgb_means
        )
        self.val_generator = ImageNetDataGenerator(
            image_directory=self.val_directory,
            label_path=self.label_path,
            label_encoding_path=self.label_encoding_path,
            eigenvalues=self.eig_values,
            eigenvectors=self.eig_vectors,
            rgb_means=self.rgb_means
        )
        self.test_generator = ImageNetDataGenerator(
            image_directory=self.test_directory,
            label_path=self.label_path,
            label_encoding_path=self.label_encoding_path,
            eigenvalues=self.eig_values,
            eigenvectors=self.eig_vectors,
            rgb_means=self.rgb_means,
            batch_size=1
        )
        # Defined after calling fit.
        self.history = None

    def fit(self, epochs : int=100):
        """Fit model for desired number of epochs."""
        # Compile model.
        compile_model(self.model)

        # Fit model and save history.
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=[self.callbacks],
            steps_per_epoch=1,
            validation_steps=1
        )

        return self.history

    def predict(self, file_name: str):
        """ Create single prediction by slicing and reflecting image and
        averaging predictions over these ten slices.
        """
        input_array = image_to_input(file_name, self.eig_values, self.eig_vectors, self.rgb_means)
        predictions = self.model.predict(input_array)

        final_prediction = np.mean(predictions, axis=0)

        return final_prediction

    def predict_n(self, file_name : str, n : int):
        """ Predicts the n most likely class labels."""
        input_array = image_to_input(file_name, self.eig_values, self.eig_vectors, self.rgb_means)
        predictions = self.model.predict(input_array)

        mean_prediction = np.mean(predictions, axis=0)

        prediction_indexes = np.argpartition(mean_prediction, -n)[-n:]

        return prediction_indexes
