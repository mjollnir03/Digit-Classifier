import numpy as np
import matplotlib.pyplot as plt


def plot_image_grid(images, labels, grid_size):
    '''
    Display multiple training images in a grid layout with their labels.
    
    Args:
        images: Array of image vectors
        labels: Array of corresponding labels
        grid_size: Tuple of (rows, cols) for the grid layout
    '''
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                # Reshape and display image
                image = images[idx].reshape((28, 28))
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].set_title(f'Label: {labels[idx]}')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load converted data from npz
npz = np.load('data/mnist.npz')
train_images = npz['training_data_images']
train_labels = npz['training_data_labels']
val_images = npz['validation_data_images']
val_labels = npz['validation_data_labels']
test_images = npz['test_data_images']
test_labels = npz['test_data_labels']

# Verify data shape
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)

# Display first 100 training images with their labels
plot_image_grid(train_images[:100], train_labels[:100], grid_size=(10,10))
