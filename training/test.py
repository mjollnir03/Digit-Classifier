import numpy as np
import os
from train import load_network

# Check if model file exists
model_path = 'models\\digit_classifier.json'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please train a model first by running main.py")
    exit(1)

# Load the trained network
print("Loading trained model...")
network = load_network(model_path)
print(f"Model loaded successfully. Architecture: {network.layer_sizes}")

# Check if data file exists
data_path = 'data/mnist.npz'
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit(1)

# Load test data
print("Loading test data...")
with np.load(data_path) as data:
    test_images = data['test_data_images']
    test_labels = data['test_data_labels']
    
    # Verify data format
    if test_images.shape[1] != 784:
        print(f"Error: Expected 784 features, got {test_images.shape[1]}")
        exit(1)
    
    if len(test_images) != len(test_labels):
        print("Error: Number of images and labels don't match")
        exit(1)

print(f"Data loaded: {len(test_images)} test samples")

# Test one sample
print("\n" + "=" * 60)
print("Testing single sample:")
print("=" * 60)
sample_index = 0
image = test_images[sample_index].reshape(784, 1)
true_label = test_labels[sample_index]

# Get network output
output = network.feed_forward(image)
predicted_label = np.argmax(output)
confidence = output[predicted_label, 0]

# Print results
print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.4f}")
print(f"Correct: {true_label == predicted_label}")

# Test accuracy on full test set
print("\n" + "=" * 60)
print("Testing full test set (10,000 samples):")
print("=" * 60)
correct = 0
for i in range(len(test_images)):
    image = test_images[i].reshape(784, 1)
    output = network.feed_forward(image)
    predicted = np.argmax(output)
    if predicted == test_labels[i]:
        correct += 1

accuracy_pct = (correct / len(test_images)) * 100
print(f"Accuracy: {correct}/{len(test_images)} ({accuracy_pct:.2f}%)")

# Show a few sample predictions
print("\n" + "=" * 60)
print("Sample predictions (first 5 test images):")
print("=" * 60)
for i in range(5):
    image = test_images[i].reshape(784, 1)
    output = network.feed_forward(image)
    predicted = np.argmax(output)
    confidence = output[predicted, 0]
    true_label = test_labels[i]
    
    status = "CORRECT" if predicted == true_label else "WRONG"
    print(f"[{status}] Image {i}: True={true_label}, Predicted={predicted}, Confidence={confidence:.4f}")

