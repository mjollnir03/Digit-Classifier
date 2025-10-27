import numpy as np
import matplotlib.pyplot as plt


def plot_image_grid(image_vector, size=(28, 28)):
    '''
    Display a single training image.
    '''
    image = image_vector.reshape(size)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


# Load converted data from npz
npz = np.load('data/mnist.npz')
train_images = npz['training_data_images']
train_labels = npz['training_data_labels']
val_images = npz['validation_data_images']
val_labels = npz['validation_data_labels']
test_images = npz['test_data_images']
test_labels = npz['test_data_labels']

# Verify data shape and display first sample
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("First label:", train_labels[0])

# Plot the first training image
plot_image_grid(train_images[0])
