import numpy as np
import os
from datetime import datetime
from train import NeuralNetwork, CrossEntropyCost, to_categorical

# Load MNIST Dataset from Compressed NumPy Archive
print("Loading MNIST dataset from data/mnist.npz...")
npz_file = np.load('data/mnist.npz')

# Extract training set (50,000 samples of 784-pixel images + labels)
training_images = npz_file['training_data_images']
training_labels = npz_file['training_data_labels']

# Extract validation set (10,000 samples for monitoring during training)
validation_images = npz_file['validation_data_images']
validation_labels = npz_file['validation_data_labels']

# Extract test set (10,000 samples for final evaluation)
test_images = npz_file['test_data_images']
test_labels = npz_file['test_data_labels']

# Data Preparation Functions
def prepare_training_data(images, labels):
    """
    Prepare training data by reshaping images into column vectors and
    converting labels to one-hot encoded format.
    
    Args:
        images: array of shape (N, 784) containing flattened 28x28 images
        labels: array of shape (N,) containing integer class labels 0-9
        
    Returns:
        list of (input_column_vector, one_hot_label) tuples ready for training
    """
    training_data = []
    
    for image_row, label_int in zip(images, labels):
        # Reshape from (784,) row vector to (784, 1) column vector
        # This format is required for matrix multiplication in the network
        image_column_vector = image_row.reshape(784, 1)
        
        # Convert integer label (e.g., 3) to one-hot vector (e.g., [0,0,0,1,0,0,0,0,0,0])
        one_hot_label = to_categorical(label_int)
        
        # Append tuple of (input, target) for this sample
        training_data.append((image_column_vector, one_hot_label))
    
    return training_data

def prepare_validation_data(images, labels):
    """
    Prepare validation/test data by reshaping images into column vectors.
    Labels are kept as integers (not one-hot) for accuracy computation.
    
    Args:
        images: array of shape (N, 784) containing flattened 28x28 images
        labels: array of shape (N,) containing integer class labels 0-9
        
    Returns:
        list of (input_column_vector, integer_label) tuples ready for evaluation
    """
    validation_data = []
    
    for image_row, label_int in zip(images, labels):
        # Reshape from (784,) row vector to (784, 1) column vector
        # This format is required for matrix multiplication in the network
        image_column_vector = image_row.reshape(784, 1)
        
        # Keep label as integer for direct comparison during accuracy evaluation
        # (accuracy function will compare argmax(prediction) to this integer)
        validation_data.append((image_column_vector, label_int))
    
    return validation_data

# Prepare All Dataset Splits
print("Preparing training data...")
training_data = prepare_training_data(training_images, training_labels)

print("Preparing validation data...")
validation_data = prepare_validation_data(validation_images, validation_labels)

print("Preparing test data...")
test_data = prepare_validation_data(test_images, test_labels)

# Initialize Neural Network
print("\nInitializing neural network...")
# Architecture: 784 input -> 128 hidden (ReLU) -> 64 hidden (ReLU) -> 10 output (sigmoid)
# Using ReLU activation for better gradient flow and Adam optimizer for faster convergence
network = NeuralNetwork([784, 128, 64, 10], cost=CrossEntropyCost, activation='relu', use_adam=True)

# Train the Network
print("Starting neural network training...")
print("-" * 60)

# Run stochastic gradient descent with improved hyperparameters
evaluation_costs, evaluation_accuracies, training_costs, training_accuracies = network.stochastic_gradient_descent(
    training_data=training_data,
    epochs=30,
    mini_batch_size=32,
    learning_rate=0.001,  # Lower learning rate for Adam
    regularization_param=5.0,
    dropout_rate=0.2,  # 20% dropout for regularization
    lr_decay=0.01,  # Small learning rate decay
    early_stopping_patience=5,  # Stop if no improvement for 5 epochs
    evaluation_data=validation_data,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True
)

# Save Trained Model with backup protection
print("-" * 60)
model_path = 'models/digit_classifier.json'

# If model exists, create a backup with timestamp
if os.path.exists(model_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'models/digit_classifier_backup_{timestamp}.json'
    
    print(f"\nExisting model found. Creating backup: {backup_path}")
    import shutil
    shutil.copy(model_path, backup_path)

print(f"\nSaving the trained model to {model_path}...")
network.save(model_path)
print("Model saved successfully.")

# Evaluate on Test Set
print("\nEvaluating the model on test data...")
correct_test_predictions = network.accuracy(test_data)
total_test_samples = len(test_data)
test_accuracy_percentage = (correct_test_predictions / total_test_samples) * 100

print(f"Test accuracy: {correct_test_predictions} / {total_test_samples} ({test_accuracy_percentage:.2f}%)")