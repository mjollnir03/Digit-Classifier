import numpy as np
import pickle
import gzip

print("Loading MNIST data from compressed pickle file...")
# Load the MNIST dataset from the gzipped pickle file.
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    # The pickle file contains training, validation, and test data, which are loaded into respective variables.
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
print("Data loaded successfully.")

# Unpack the loaded data into separate arrays for images and labels.
training_images, training_labels = training_data
validation_images, validation_labels = validation_data
test_images, test_labels = test_data

print("Saving data to a compressed NPZ file...")
# Save the unpacked data into a compressed .npz file for efficient storage and access.
# The data is saved with descriptive keys for easy identification.
np.savez_compressed('data/mnist.npz',
    training_data_images=training_images,
    training_data_labels=training_labels,
    validation_data_images=validation_images,
    validation_data_labels=validation_labels,
    test_data_images=test_images,
    test_data_labels=test_labels
)
print("NPZ file created successfully at 'data/mnist.npz'.")
