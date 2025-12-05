import json
import random
import sys
import numpy as np

# Activation Functions
def sigmoid(z):
    """
    Sigmoid activation function: 1 / (1 + e^(-z))
    Numerically stable implementation.
    """
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def sigmoid_prime(z):
    """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU activation function: max(0, z)"""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    return (z > 0).astype(float)

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
    Supports L2 regularization, dropout, multiple activation functions, and Adam optimizer.
    """
    def __init__(self, layer_sizes, cost=CrossEntropyCost, activation='sigmoid', use_adam=False):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: list of integers specifying neurons per layer (e.g., [784, 128, 64, 10])
            cost: cost class to use (QuadraticCost or CrossEntropyCost)
            activation: 'sigmoid' or 'relu' for hidden layers (output always uses sigmoid)
            use_adam: whether to use Adam optimizer instead of vanilla SGD
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.cost = cost
        self.activation = activation
        self.use_adam = use_adam
        
        # Set activation functions
        if activation == 'relu':
            self.hidden_activation = relu
            self.hidden_activation_prime = relu_prime
        else:
            self.hidden_activation = sigmoid
            self.hidden_activation_prime = sigmoid_prime
            
        self.initialize_weights()
        
        # Initialize Adam optimizer parameters if needed
        if use_adam:
            self.m_weights = [np.zeros(w.shape) for w in self.weights]
            self.v_weights = [np.zeros(w.shape) for w in self.weights]
            self.m_biases = [np.zeros(b.shape) for b in self.biases]
            self.v_biases = [np.zeros(b.shape) for b in self.biases]
            self.adam_t = 0  # Time step for Adam

    def initialize_weights(self):
        """
        Initialize biases and weights using He/Xavier initialization.
        He initialization for ReLU, Xavier for sigmoid.
        """
        # Initialize biases to small random values
        self.biases = [
            np.random.randn(num_neurons, 1) * 0.01
            for num_neurons in self.layer_sizes[1:]
        ]
        
        # Initialize weights with appropriate scaling
        self.weights = []
        for i, (input_size, output_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            # Use He initialization for ReLU, Xavier for sigmoid
            if self.activation == 'relu' and i < len(self.layer_sizes) - 2:
                # He initialization: scale by sqrt(2/fan_in)
                scale = np.sqrt(2.0 / input_size)
            else:
                # Xavier initialization: scale by sqrt(1/fan_in)
                scale = np.sqrt(1.0 / input_size)
            
            self.weights.append(np.random.randn(output_size, input_size) * scale)

    def feed_forward(self, activation, dropout_rate=0.0, training=False):
        """
        Propagate input through network and return output activation.
        
        Args:
            activation: input column vector of shape (784, 1)
            dropout_rate: probability of dropping neurons (0.0 to 1.0)
            training: whether in training mode (applies dropout)
            
        Returns:
            output activation of shape (num_output_neurons, 1)
        """
        # Apply hidden layers with chosen activation
        for i, (bias, weight) in enumerate(zip(self.biases[:-1], self.weights[:-1])):
            weighted_sum = np.dot(weight, activation) + bias
            activation = self.hidden_activation(weighted_sum)
            
            # Apply dropout during training
            if training and dropout_rate > 0.0:
                mask = np.random.binomial(1, 1 - dropout_rate, size=activation.shape)
                activation = activation * mask / (1 - dropout_rate)
        
        # Output layer always uses sigmoid for probability outputs
        weighted_sum = np.dot(self.weights[-1], activation) + self.biases[-1]
        activation = sigmoid(weighted_sum)
        
        return activation

    def stochastic_gradient_descent(
        self, 
        training_data, 
        epochs, 
        mini_batch_size, 
        learning_rate,
        regularization_param=0.0,
        dropout_rate=0.0,
        lr_decay=0.0,
        early_stopping_patience=None,
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
            learning_rate: initial learning rate (eta) for weight updates
            regularization_param: L2 regularization strength (lambda)
            dropout_rate: probability of dropping neurons (0.0 to 1.0)
            lr_decay: learning rate decay factor per epoch
            early_stopping_patience: stop if no improvement for N epochs
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
        
        # Early stopping variables
        best_accuracy = 0
        patience_counter = 0
        best_weights = None
        best_biases = None

        # Main training loop
        current_lr = learning_rate
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
                    current_lr, 
                    regularization_param, 
                    num_training_samples,
                    dropout_rate
                )
            
            print(f"Epoch {epoch} training complete")

            # Monitor training cost
            if monitor_training_cost:
                training_cost = self.total_cost(training_data, regularization_param)
                training_costs.append(training_cost)
                print(f"Cost on training data: {training_cost:.4f}")
            
            # Monitor training accuracy
            if monitor_training_accuracy:
                correct_train_predictions = self.accuracy(training_data, convert=True)
                training_accuracies.append(correct_train_predictions)
                train_acc_pct = (correct_train_predictions / num_training_samples) * 100
                print(f"Accuracy on training data: {correct_train_predictions} / {num_training_samples} ({train_acc_pct:.2f}%)")

            # Monitor evaluation cost
            if monitor_evaluation_cost:
                evaluation_cost = self.total_cost(evaluation_data, regularization_param, convert=True)
                evaluation_costs.append(evaluation_cost)
                print(f"Cost on evaluation data: {evaluation_cost:.4f}")

            # Monitor evaluation accuracy
            if monitor_evaluation_accuracy:
                correct_eval_predictions = self.accuracy(evaluation_data)
                evaluation_accuracies.append(correct_eval_predictions)
                eval_acc_pct = (correct_eval_predictions / num_evaluation_samples) * 100
                print(f"Accuracy on evaluation data: {correct_eval_predictions} / {num_evaluation_samples} ({eval_acc_pct:.2f}%)")
                
                # Early stopping check
                if early_stopping_patience is not None:
                    if correct_eval_predictions > best_accuracy:
                        best_accuracy = correct_eval_predictions
                        patience_counter = 0
                        # Save best weights
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        print(f"New best accuracy! Resetting patience counter.")
                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                        
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            # Restore best weights
                            if best_weights is not None:
                                self.weights = best_weights
                                self.biases = best_biases
                                print("Restored best weights")
                            break
            
            # Learning rate decay
            if lr_decay > 0:
                current_lr = learning_rate / (1 + lr_decay * epoch)
                print(f"Learning rate: {current_lr:.6f}")
            
            print()

        return evaluation_costs, evaluation_accuracies, training_costs, training_accuracies

    def update_mini_batch(self, mini_batch, learning_rate, regularization_param, num_training_samples, dropout_rate=0.0):
        """
        Update network weights and biases using one mini-batch of training data.
        
        Args:
            mini_batch: list of (x, y) training samples
            learning_rate: learning rate (eta)
            regularization_param: L2 regularization strength (lambda)
            num_training_samples: total number of training samples (for L2 scaling)
            dropout_rate: probability of dropping neurons during training
        """
        # Initialize gradient accumulators
        accumulated_bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        accumulated_weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        # Accumulate gradients over all samples in mini-batch
        for sample_input, sample_output in mini_batch:
            delta_bias_gradients, delta_weight_gradients = self.back_propagate(
                sample_input, sample_output, dropout_rate
            )
            
            # Add to accumulated gradients
            accumulated_bias_gradients = [
                accum_grad + delta_grad 
                for accum_grad, delta_grad in zip(accumulated_bias_gradients, delta_bias_gradients)
            ]
            accumulated_weight_gradients = [
                accum_grad + delta_grad 
                for accum_grad, delta_grad in zip(accumulated_weight_gradients, delta_weight_gradients)
            ]

        mini_batch_size = len(mini_batch)
        
        if self.use_adam:
            # Adam optimizer update
            self.adam_t += 1
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            
            # Update weights
            for i in range(len(self.weights)):
                # Average gradient over mini-batch
                gradient = accumulated_weight_gradients[i] / mini_batch_size
                
                # Add L2 regularization gradient
                gradient += (regularization_param / num_training_samples) * self.weights[i]
                
                # Update biased first moment estimate
                self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * gradient
                
                # Update biased second raw moment estimate
                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (gradient ** 2)
                
                # Compute bias-corrected estimates
                m_hat = self.m_weights[i] / (1 - beta1 ** self.adam_t)
                v_hat = self.v_weights[i] / (1 - beta2 ** self.adam_t)
                
                # Update weights
                self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Update biases
            for i in range(len(self.biases)):
                gradient = accumulated_bias_gradients[i] / mini_batch_size
                
                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * gradient
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (gradient ** 2)
                
                m_hat = self.m_biases[i] / (1 - beta1 ** self.adam_t)
                v_hat = self.v_biases[i] / (1 - beta2 ** self.adam_t)
                
                self.biases[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            # Standard SGD update with L2 regularization
            regularization_factor = 1 - learning_rate * (regularization_param / num_training_samples)
            
            self.weights = [
                regularization_factor * weight - (learning_rate / mini_batch_size) * weight_gradient
                for weight, weight_gradient in zip(self.weights, accumulated_weight_gradients)
            ]
            
            self.biases = [
                bias - (learning_rate / mini_batch_size) * bias_gradient
                for bias, bias_gradient in zip(self.biases, accumulated_bias_gradients)
            ]

    def back_propagate(self, sample_input, sample_output, dropout_rate=0.0):
        """
        Compute gradients for a single training sample using backpropagation.
        
        Args:
            sample_input: input column vector x of shape (784, 1)
            sample_output: target output (one-hot for training, int label for others)
            dropout_rate: probability of dropping neurons during training
            
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
        dropout_masks = []

        # Hidden layers
        for i, (layer_bias, layer_weight) in enumerate(zip(self.biases[:-1], self.weights[:-1])):
            weighted_sum = np.dot(layer_weight, current_activation) + layer_bias
            all_z_values.append(weighted_sum)
            current_activation = self.hidden_activation(weighted_sum)
            
            # Apply dropout
            if dropout_rate > 0.0:
                mask = np.random.binomial(1, 1 - dropout_rate, size=current_activation.shape)
                current_activation = current_activation * mask / (1 - dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
            
            all_activations.append(current_activation)
        
        # Output layer (always sigmoid)
        weighted_sum = np.dot(self.weights[-1], current_activation) + self.biases[-1]
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
        current_delta = output_delta
        for layer_index in range(2, self.num_layers):
            current_z = all_z_values[-layer_index]
            activation_derivative = self.hidden_activation_prime(current_z)
            
            # Propagate delta from next layer
            next_layer_weight = self.weights[-layer_index + 1]
            current_delta = np.dot(next_layer_weight.transpose(), current_delta) * activation_derivative
            
            # Apply dropout mask if used
            mask_idx = len(dropout_masks) - layer_index + 1
            if mask_idx >= 0 and mask_idx < len(dropout_masks) and dropout_masks[mask_idx] is not None:
                current_delta = current_delta * dropout_masks[mask_idx] / (1 - dropout_rate)
            
            # Compute gradients for current layer
            bias_gradients[-layer_index] = current_delta
            previous_activation = all_activations[-layer_index - 1]
            weight_gradients[-layer_index] = np.dot(current_delta, previous_activation.transpose())
            
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
            "cost": str(self.cost.__name__),
            "activation": self.activation
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
    
    # Get activation function (default to sigmoid for backward compatibility)
    activation = network_data.get("activation", "sigmoid")
    
    # Create new network with same architecture
    network = NeuralNetwork(network_data["layers"], cost=cost_class, activation=activation)
    
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
