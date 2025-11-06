import json
import random
import sys
import numpy as np

# Cost Functions
class QuadraticCost:
    """
    Quadratic cost function used for regression tasks.
    Cost = 0.5 * ||activation - y||^2
    """
    @staticmethod
    def function(activation, y):
        """Compute quadratic cost for a single sample."""
        difference = activation - y
        squared_error = np.linalg.norm(difference) ** 2
        cost = 0.5 * squared_error
        return cost

    @staticmethod
    def delta(z, activation, y):
        """Compute output layer error delta for backpropagation."""
        output_error = activation - y
        delta = output_error * sigmoid_prime(z)
        return delta

class CrossEntropyCost:
    """
    Cross-entropy cost function for classification tasks.
    Numerically stable with sigmoid outputs.
    """
    @staticmethod
    def function(activation, y):
        """Compute cross-entropy cost for a single sample."""
        # Use nan_to_num to handle log(0) gracefully
        positive_class_term = -y * np.log(activation)
        negative_class_term = -(1 - y) * np.log(1 - activation)
        total_cost = np.sum(positive_class_term + negative_class_term)
        return np.nan_to_num(total_cost)

    @staticmethod
    def delta(z, activation, y):
        """Compute output layer error delta for backpropagation."""
        # For cross-entropy + sigmoid, delta simplifies to (activation - y)
        delta = activation - y
        return delta

# Neural Network
class NeuralNetwork:
    """
    A fully-connected feedforward neural network trained with stochastic gradient descent.
    Supports L2 regularization and configurable cost functions.
    """
    def __init__(self, layer_sizes, cost=CrossEntropyCost):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: list of integers specifying neurons per layer (e.g., [784, 128, 64, 10])
            cost: cost class to use (QuadraticCost or CrossEntropyCost)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.cost = cost
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize biases and weights using He/Xavier initialization.
        Weights scaled by 1/sqrt(fan_in) to stabilize training.
        """
        # Initialize biases: one column vector per hidden/output layer
        self.biases = [
            np.random.randn(num_neurons, 1) 
            for num_neurons in self.layer_sizes[1:]
        ]
        
        # Initialize weights: one matrix per layer transition
        # Shape: (output_size, input_size)
        self.weights = [
            np.random.randn(output_size, input_size) / np.sqrt(input_size)
            for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def feed_forward(self, activation):
        """
        Propagate input through network and return output activation.
        
        Args:
            activation: input column vector of shape (784, 1)
            
        Returns:
            output activation of shape (num_output_neurons, 1)
        """
        # Apply each layer: z = W*a + b, then a = sigmoid(z)
        for bias, weight in zip(self.biases, self.weights):
            weighted_sum = np.dot(weight, activation) + bias
            activation = sigmoid(weighted_sum)
        return activation

    def stochastic_gradient_descent(
        self, 
        training_data, 
        epochs, 
        mini_batch_size, 
        learning_rate,
        regularization_param=0.0, 
        evaluation_data=None,
        monitor_evaluation_cost=False, 
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False, 
        monitor_training_accuracy=False
    ):
        """
        Train the network using stochastic gradient descent with mini-batches.
        
        Args:
            training_data: list of (x, y) tuples for training
            epochs: number of passes over training data
            mini_batch_size: number of samples per gradient update
            learning_rate: learning rate (eta) for weight updates
            regularization_param: L2 regularization strength (lambda)
            evaluation_data: optional (x, y) tuples for validation monitoring
            monitor_*: flags to enable printing of costs/accuracies
            
        Returns:
            tuple of (eval_costs, eval_accuracies, train_costs, train_accuracies) lists
        """
        num_training_samples = len(training_data)
        
        # Determine evaluation set size if provided
        if evaluation_data:
            num_evaluation_samples = len(evaluation_data)
        
        # Initialize monitoring lists
        evaluation_costs = []
        evaluation_accuracies = []
        training_costs = []
        training_accuracies = []

        # Main training loop
        for epoch in range(epochs):
            # Shuffle training data for each epoch
            random.shuffle(training_data)
            
            # Split into mini-batches
            mini_batches = [
                training_data[batch_start:batch_start + mini_batch_size]
                for batch_start in range(0, num_training_samples, mini_batch_size)
            ]
            
            # Update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, 
                    learning_rate, 
                    regularization_param, 
                    num_training_samples
                )
            
            print(f"Epoch {epoch} training complete")

            # Monitor training cost
            if monitor_training_cost:
                training_cost = self.total_cost(training_data, regularization_param)
                training_costs.append(training_cost)
                print(f"Cost on training data: {training_cost}")
            
            # Monitor training accuracy
            if monitor_training_accuracy:
                correct_train_predictions = self.accuracy(training_data, convert=True)
                training_accuracies.append(correct_train_predictions)
                print(f"Accuracy on training data: {correct_train_predictions} / {num_training_samples}")

            # Monitor evaluation cost
            if monitor_evaluation_cost:
                evaluation_cost = self.total_cost(evaluation_data, regularization_param, convert=True)
                evaluation_costs.append(evaluation_cost)
                print(f"Cost on evaluation data: {evaluation_cost}")

            # Monitor evaluation accuracy
            if monitor_evaluation_accuracy:
                correct_eval_predictions = self.accuracy(evaluation_data)
                evaluation_accuracies.append(correct_eval_predictions)
                print(f"Accuracy on evaluation data: {correct_eval_predictions} / {num_evaluation_samples}")
            
            print()

        return evaluation_costs, evaluation_accuracies, training_costs, training_accuracies

    def update_mini_batch(self, mini_batch, learning_rate, regularization_param, num_training_samples):
        """
        Update network weights and biases using one mini-batch of training data.
        
        Args:
            mini_batch: list of (x, y) training samples
            learning_rate: learning rate (eta)
            regularization_param: L2 regularization strength (lambda)
            num_training_samples: total number of training samples (for L2 scaling)
        """
        # Initialize gradient accumulators
        accumulated_bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        accumulated_weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        # Accumulate gradients over all samples in mini-batch
        for sample_input, sample_output in mini_batch:
            delta_bias_gradients, delta_weight_gradients = self.back_propagate(sample_input, sample_output)
            
            # Add to accumulated gradients
            accumulated_bias_gradients = [
                accum_grad + delta_grad 
                for accum_grad, delta_grad in zip(accumulated_bias_gradients, delta_bias_gradients)
            ]
            accumulated_weight_gradients = [
                accum_grad + delta_grad 
                for accum_grad, delta_grad in zip(accumulated_weight_gradients, delta_weight_gradients)
            ]

        # Update weights with L2 regularization
        # w = (1 - eta * lambda / N) * w - (eta / m) * nabla_w
        mini_batch_size = len(mini_batch)
        regularization_factor = 1 - learning_rate * (regularization_param / num_training_samples)
        
        self.weights = [
            regularization_factor * weight - (learning_rate / mini_batch_size) * weight_gradient
            for weight, weight_gradient in zip(self.weights, accumulated_weight_gradients)
        ]
        
        # Update biases (no regularization on biases)
        # b = b - (eta / m) * nabla_b
        self.biases = [
            bias - (learning_rate / mini_batch_size) * bias_gradient
            for bias, bias_gradient in zip(self.biases, accumulated_bias_gradients)
        ]

    def back_propagate(self, sample_input, sample_output):
        """
        Compute gradients for a single training sample using backpropagation.
        
        Args:
            sample_input: input column vector x of shape (784, 1)
            sample_output: target output (one-hot for training, int label for others)
            
        Returns:
            tuple of (bias_gradients, weight_gradients) lists
        """
        # Initialize gradient accumulators
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
        
        # Forward pass: store all activations and z-values
        current_activation = sample_input
        all_activations = [sample_input]
        all_z_values = []

        for layer_bias, layer_weight in zip(self.biases, self.weights):
            weighted_sum = np.dot(layer_weight, current_activation) + layer_bias
            all_z_values.append(weighted_sum)
            current_activation = sigmoid(weighted_sum)
            all_activations.append(current_activation)

        # Backward pass: compute output layer error delta
        output_z = all_z_values[-1]
        output_activation = all_activations[-1]
        output_delta = self.cost.delta(output_z, output_activation, sample_output)
        
        # Compute gradients for output layer
        bias_gradients[-1] = output_delta
        previous_activation = all_activations[-2]
        weight_gradients[-1] = np.dot(output_delta, previous_activation.transpose())

        # Backpropagate error through hidden layers
        for layer_index in range(2, self.num_layers):
            current_z = all_z_values[-layer_index]
            sigmoid_derivative = sigmoid_prime(current_z)
            
            # Propagate delta from next layer
            next_layer_weight = self.weights[-layer_index + 1]
            previous_delta = output_delta  # Use output_delta as it accumulates through loop
            current_delta = np.dot(next_layer_weight.transpose(), previous_delta) * sigmoid_derivative
            
            # Compute gradients for current layer
            bias_gradients[-layer_index] = current_delta
            previous_activation = all_activations[-layer_index - 1]
            weight_gradients[-layer_index] = np.dot(current_delta, previous_activation.transpose())
            
            # Update output_delta for next iteration
            output_delta = current_delta
            
        return bias_gradients, weight_gradients

    def accuracy(self, data, convert=False):
        """
        Compute number of correct predictions on a dataset.
        
        Args:
            data: list of (x, y) tuples
            convert: if True, y is one-hot encoded (use argmax); else y is integer label
            
        Returns:
            count of correct predictions (not percentage)
        """
        correct_count = 0
        
        for sample_input, sample_output in data:
            # Get predicted class
            output_activation = self.feed_forward(sample_input)
            predicted_class = np.argmax(output_activation)
            
            # Get true class
            if convert:
                # sample_output is one-hot; convert to integer
                true_class = np.argmax(sample_output)
            else:
                # sample_output is already an integer
                true_class = sample_output
            
            # Check if prediction matches true label
            if predicted_class == true_class:
                correct_count += 1
        
        return correct_count

    def total_cost(self, data, regularization_param, convert=False):
        """
        Compute average cost over a dataset plus L2 regularization term.
        
        Args:
            data: list of (x, y) tuples
            regularization_param: L2 regularization strength (lambda)
            convert: if True, convert integer labels to one-hot
            
        Returns:
            total cost (average sample cost + L2 penalty)
        """
        total_sample_cost = 0.0
        num_samples = len(data)
        
        # Sum cost across all samples
        for sample_input, sample_output in data:
            output_activation = self.feed_forward(sample_input)
            
            # Convert label if needed
            if convert:
                sample_output = to_categorical(sample_output)
            
            sample_cost = self.cost.function(output_activation, sample_output)
            total_sample_cost += sample_cost / num_samples
        
        # Add L2 regularization penalty
        weight_sum_of_squares = sum(np.linalg.norm(weight) ** 2 for weight in self.weights)
        l2_penalty = 0.5 * (regularization_param / num_samples) * weight_sum_of_squares
        
        total_cost = total_sample_cost + l2_penalty
        return total_cost

    def save(self, filename):
        """
        Save the trained network to a JSON file.
        
        Args:
            filename: path to output JSON file
        """
        network_data = {
            "layers": self.layer_sizes,
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
            "cost": str(self.cost.__name__)
        }
        
        with open(filename, "w") as output_file:
            json.dump(network_data, output_file)

# Loading and Helper Functions
def load_network(filename):
    """
    Load a trained network from a JSON file.
    
    Args:
        filename: path to JSON file
        
    Returns:
        reconstructed NeuralNetwork instance with saved weights and biases
    """
    with open(filename, "r") as input_file:
        network_data = json.load(input_file)
    
    # Reconstruct cost function class from saved name
    cost_class_name = network_data["cost"]
    cost_class = getattr(sys.modules[__name__], cost_class_name)
    
    # Create new network with same architecture
    network = NeuralNetwork(network_data["layers"], cost=cost_class)
    
    # Restore saved weights and biases as NumPy arrays
    network.weights = [np.array(weight_list) for weight_list in network_data["weights"]]
    network.biases = [np.array(bias_list) for bias_list in network_data["biases"]]
    
    return network

def to_categorical(label_index):
    """
    Convert integer label to one-hot encoded column vector.
    
    Args:
        label_index: integer class label (0-9 for MNIST)
        
    Returns:
        one-hot column vector of shape (10, 1)
    """
    one_hot_vector = np.zeros((10, 1))
    one_hot_vector[label_index] = 1.0
    return one_hot_vector

def sigmoid(z):
    """
    Sigmoid activation function: 1 / (1 + e^(-z))
    
    Args:
        z: input value or array
        
    Returns:
        activated output in range (0, 1)
    """
    activated_output = 1.0 / (1.0 + np.exp(-z))
    return activated_output

def sigmoid_prime(z):
    """
    Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))
    
    Args:
        z: input value or array
        
    Returns:
        derivative value
    """
    sigmoid_value = sigmoid(z)
    derivative = sigmoid_value * (1 - sigmoid_value)
    return derivative
