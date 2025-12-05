import numpy as np
from train import load_network

# Load the trained network
network = load_network('models/digit_classifier.json')

# Load test data
with np.load('data/mnist.npz') as data:
    test_images = data['test_data_images']
    test_labels = data['test_data_labels']

# Test one sample
sample_index = 0
image = test_images[sample_index].reshape(784, 1)  # Reshape to column vector
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

# Test accuracy on first 100 samples
correct = 0
for i in range(10000):
    image = test_images[i].reshape(784, 1)
    output = network.feed_forward(image)
    predicted = np.argmax(output)
    if predicted == test_labels[i]:
        correct += 1

print(f"\nAccuracy on 100 samples: {correct}/100 ({correct}%)")
