import numpy as np
import matplotlib.pyplot as plt

def plot_image_grid(images, labels, grid_size):
    """
    Displays a grid of images with their corresponding labels.
    
    Args:
        images (np.ndarray): A collection of images to be displayed.
        labels (np.ndarray): The labels corresponding to the images.
        grid_size (tuple): A tuple specifying the number of rows and columns for the grid.
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                # Reshape the image to 28x28 pixels and display it
                image = images[index].reshape((28, 28))
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].set_title(f'Label: {labels[index]}')
                axes[i, j].axis('off')
            else:
                # Hide axes for empty subplots
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load the dataset from the .npz file
with np.load('data/mnist.npz') as data:
    training_images = data['training_data_images']
    training_labels = data['training_data_labels']
    validation_images = data['validation_data_images']
    validation_labels = data['validation_data_labels']
    test_images = data['test_data_images']
    test_labels = data['test_data_labels']

# Verify the shapes of the loaded data
print(f"Training images shape: {training_images.shape}")
print(f"Training labels shape: {training_labels.shape}")

# Display the first 100 training images in a 10x10 grid
plot_image_grid(training_images[:100], training_labels[:100], grid_size=(10, 10))
