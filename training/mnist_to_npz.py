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

'''
# Load converted data from npz
npz = np.load('data/mnist.npz')
train_images = npz['training_data_images']
train_labels = npz['training_data_labels']
val_images = npz['validation_data_images']
val_labels = npz['validation_data_labels']
test_images = npz['test_data_images']
test_labels = npz['test_data_labels']

# Compare the data arrays
print("Verification")
print("Train images match:", np.array_equal(training_data[0], train_images))
print("Train labels match:", np.array_equal(training_data[1], train_labels))
print("Val images match:", np.array_equal(validation_data[0], val_images))
print("Val labels match:", np.array_equal(validation_data[1], val_labels))
print("Test images match:", np.array_equal(test_data[0], test_images))
print("Test labels match:", np.array_equal(test_data[1], test_labels))

'''