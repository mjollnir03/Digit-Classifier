import numpy as np
import json
import pickle as pkl
import gzip


class Neural_Network():

    def __init__(self):
        pass

    def feed_forward(self):
        pass


    def stochastic_gradient_descent(self):
        pass


    def back_propagation(self):
        pass

    def cost_function(self):
        pass







# TESTING

# Load MNIST data from compressed pickle
with gzip.open('data/mnist.pkl.gz', 'rb') as file:
    training_data, validation_data, test_data = pkl.load(file, encoding='latin1')

# Unpack data for clarity
X_train, y_train = training_data
X_val, y_val = validation_data
X_test, y_test = test_data

# Display basic information
print("MNIST Dataset Info")
print(f'Training samples: {X_train.shape}, Labels: {y_train.shape}')
print(f'Validation samples: {X_val.shape}, Labels: {y_val.shape}')
print(f'Test samples: {X_test.shape}, Labels: {y_test.shape}')
print(f'Each image has {X_train.shape[1]} pixels (flattened 28x28).')
print(f'Data type of a single sample: {type(X_train[0])}')
print()

# Save as NumPy Zip format
np.savez('data/mnist.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test)

print("MNIST data saved as 'data/mnist.npz'.")


# Step 1: Load original MNIST from .pkl.gz
with gzip.open('data/mnist.pkl.gz', 'rb') as file:
    orig_train, orig_val, orig_test = pkl.load(file, encoding='latin1')

X_train_orig, y_train_orig = orig_train
X_val_orig, y_val_orig = orig_val
X_test_orig, y_test_orig = orig_test

# Step 2: Load saved .npz file
npz_data = np.load('data/mnist.npz')

X_train = npz_data['X_train']
y_train = npz_data['y_train']
X_val = npz_data['X_val']
y_val = npz_data['y_val']
X_test = npz_data['X_test']
y_test = npz_data['y_test']

# Step 3: Compare arrays
print("=== Verification ===")
print("Train images match:", np.array_equal(X_train_orig, X_train))
print("Train labels match:", np.array_equal(y_train_orig, y_train))
print("Validation images match:", np.array_equal(X_val_orig, X_val))
print("Validation labels match:", np.array_equal(y_val_orig, y_val))
print("Test images match:", np.array_equal(X_test_orig, X_test))
print("Test labels match:", np.array_equal(y_test_orig, y_test))