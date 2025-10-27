import numpy as np
import pickle as pkl
import gzip

# Load MNIST data from compressed pickle
with gzip.open('data/mnist.pkl.gz', 'rb') as file:
    training_data, validation_data, test_data = pkl.load(file, encoding='latin1')

# Properly label image and label arrays
training_data_images, training_data_labels = training_data
validation_data_images, validation_data_labels = validation_data
test_data_images, test_data_labels = test_data

# Save ndarrays into a npz file using descriptive names
np.savez_compressed('data/mnist.npz',
    training_data_images=training_data_images,
    training_data_labels=training_data_labels,
    validation_data_images=validation_data_images,
    validation_data_labels=validation_data_labels,
    test_data_images=test_data_images,
    test_data_labels=test_data_labels
)
