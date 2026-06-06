# -*- coding: utf-8 -*-
# classifier.py
"""
Multi-Layer Perceptron (Neural Network) Classifier for Pacman

Author: Mark Saviour Farrugia
Course Code: [REDACTED]
Subject: Machine Learning


This classifier implements a multi-layer perceptron (MLP) neural network from scratch using only numpy. The network learns to map binary feature vectors (encoding walls, food, and ghosts around Pacman) to one of four movement actions (North, East, South, West).

Key Components:
    * Architecture: Input -> 64-neuron hidden (ReLU) -> 32-neuron hidden (ReLU) -> 4-neuron output (softmax)
    * Loss Function: Categorical cross-entropy with L2 regularisation
    * Optimiser: Adam (Adaptive Moment Estimation) for fast, stable convergence
    * Weight Initialisation: He initialisation (suited for ReLU activations)
    * Training: Mini-batch stochastic gradient descent with backpropagation
    * Early Stopping: On a held-out validation set to prevent overfitting

Design Decisions:
    * Two hidden layers provide enough capacity to learn non-linear decision boundaries without overfitting the small training set (~127 examples).
    * Adam is preferred over plain SGD because it adapts per-parameter learning rates, leading to more robust convergence with less hyperparameter tuning.
    * He initialisation (std = sqrt(2/n_in)) accounts for ReLU zeroing out roughly half the inputs, keeping gradient magnitudes stable across layers.
    * Feature dimensions are inferred from data shape at training time, so the classifier works with any feature vector length.
    * The predict() method includes a safety fallback: if the top prediction is not a legal move, it selects the highest-probability legal move.
"""

import numpy as np


class Classifier:
    """
    A multi-layer perceptron classifier for Pacman move prediction.

    Trains on binary feature vectors and predicts one of four movement
    actions (0=North, 1=East, 2=South, 3=West). 
    """

    def __init__(self):
        """Initialise the classifier with default hyperparameters."""
        # --- Network architecture ---
        self.hidden_sizes = [64, 32]  # neurons in each hidden layer
        self.n_classes = 4            # North, East, South, West

        # --- Training hyperparameters ---
        self.learning_rate = 0.001    # initial learning rate for Adam
        self.epochs = 500             # maximum training epochs
        self.batch_size = 32          # mini-batch size
        self.l2_lambda = 1e-4         # L2 regularisation strength
        self.validation_split = 0.15  # fraction held out for validation
        self.patience = 50            # early stopping patience (epochs)

        # --- Adam optimiser hyperparameters ---
        self.beta1 = 0.9              # exponential decay rate, 1st moment
        self.beta2 = 0.999            # exponential decay rate, 2nd moment
        self.epsilon = 1e-8           # small constant for numerical stability

        # --- Learned parameters (populated by fit) ---
        self.weights = []  # list of weight matrices, one per layer
        self.biases = []   # list of bias vectors, one per layer
        self.trained = False

        # --- Direction string to integer mapping for legal-move checks ---
        self._direction_to_int = {
            'North': 0, 'East': 1, 'South': 2, 'West': 3
        }

    def reset(self):
        """Reset the classifier, clearing all learned parameters."""
        self.weights = []
        self.biases = []
        self.trained = False

    def fit(self, data, target):
        """
        Train the neural network on the provided data.

        Performs mini-batch gradient descent with Adam optimisation and
        early stopping based on validation loss.

        Args:
            data:   list of lists — each inner list is a binary feature vector.
            target: list of ints — each is a class label in {0, 1, 2, 3}.
        """
        # Convert inputs to numpy arrays for vectorised computation
        X = np.array(data, dtype=np.float64)
        y = np.array(target, dtype=np.int64)

        # Derive the number of features from the data (not hardcoded)
        n_samples, n_features = X.shape

        # One-hot encode target labels for cross-entropy loss
        y_onehot = self._one_hot(y, self.n_classes)

        # Split data into training and validation sets
        X_train, y_train, X_val, y_val = self._train_val_split(
            X, y_onehot, self.validation_split
        )

        # Initialise network weights with He initialisation
        self._init_weights(n_features)

        # Initialise Adam moment estimates to zero
        m_w, v_w, m_b, v_b = self._init_adam()

        # Early stopping bookkeeping
        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        wait = 0

        # ---- Main training loop ----
        for epoch in range(1, self.epochs + 1):
            # Shuffle training data at the start of each epoch
            perm = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            # Process mini-batches
            for start in range(0, X_train.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_train.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass — compute activations at every layer
                activations, pre_activations = self._forward(X_batch)

                # Backward pass — compute gradients via backpropagation
                grad_w, grad_b = self._backward(
                    X_batch, y_batch, activations, pre_activations
                )

                # Update parameters using Adam
                self._adam_update(grad_w, grad_b, m_w, v_w, m_b, v_b, epoch)

            # ---- Early stopping check on validation set ----
            if X_val.shape[0] > 0:
                val_acts, _ = self._forward(X_val)
                val_loss = self._cross_entropy_loss(val_acts[-1], y_val)

                if val_loss < best_val_loss:
                    # Improvement found — save current weights
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    wait = 0
                else:
                    wait += 1

                # If no improvement for `patience` epochs, stop early
                if wait >= self.patience:
                    self.weights = best_weights
                    self.biases = best_biases
                    break

        # Restore best weights if training completed without early stop
        if best_weights is not None and wait < self.patience:
            self.weights = best_weights
            self.biases = best_biases

        self.trained = True

    def predict(self, data, legal=None):
        """
        Predict the best action for a single feature vector.

        Performs a forward pass through the trained network, then selects
        the highest-probability action that is also legal. Falls back to
        a random legal action if the model is untrained.

        Args:
            data:  a single feature vector (list of ints, length n_features).
            legal: a list of legal actions (Direction strings or ints), or None.

        Returns:
            An integer in {0, 1, 2, 3} representing the predicted action.
        """
        # Safety - Pick a random legal move (Last case)
        if not self.trained:
            return self._safe_random_legal(legal)

        # Reshape to 2D array for the forward pass (1 sample x n_features)
        X = np.array(data, dtype=np.float64).reshape(1, -1)

        # Forward pass to obtain class probabilities
        activations, _ = self._forward(X)
        probabilities = activations[-1].flatten()

        # Convert legal actions to integer labels
        legal_ints = self._legal_to_ints(legal)

        if legal_ints:
            # Rank actions by descending probability
            ranked_actions = np.argsort(probabilities)[::-1]
            for action in ranked_actions:
                if int(action) in legal_ints:
                    return int(action)
            # Fallback to random legal move (Last case)
            return int(np.random.choice(list(legal_ints)))

        # No legal-move information provided — return raw argmax
        return int(np.argmax(probabilities))

    def _init_weights(self, n_features):
        """
        Initialise weights with He initialisation and biases to zero.

        He init draws weights from N(0, sqrt(2/n_in)), which preserves
        gradient variance through ReLU layers during the early stages
        of training.

        Args:
            n_features: dimensionality of the input feature vectors.
        """
        self.weights = []
        self.biases = []

        # Full layer-size list: input -> hidden1 -> hidden2 -> output
        layer_sizes = [n_features] + self.hidden_sizes + [self.n_classes]

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # He initialisation standard deviation
            std = np.sqrt(2.0 / n_in)
            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def _init_adam(self):
        """
        Create zero-initialised first and second moment estimate arrays
        for the Adam optimiser, matching each weight and bias matrix.

        Returns:
            Tuple (m_w, v_w, m_b, v_b) of lists of numpy arrays.
        """
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        return m_w, v_w, m_b, v_b

    def _forward(self, X):
        """
        Propagate input X through all layers of the network.

        Hidden layers use ReLU activation; the output layer uses softmax.

        Args:
            X: input array, shape (n_samples, n_features).

        Returns:
            activations:     list of post-activation arrays per layer.
            pre_activations: list of pre-activation (z = XW + b) arrays.
        """
        activations = []
        pre_activations = []
        current = X

        for i in range(len(self.weights)):
            # Linear transformation
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            if i < len(self.weights) - 1:
                # ReLU activation - Hidden layer
                current = self._relu(z)
            else:
                # softmax activation - Output layer
                current = self._softmax(z)

            activations.append(current)

        return activations, pre_activations

    def _backward(self, X, y_onehot, activations, pre_activations):
        """
        Compute gradients of the loss with respect to all weights and biases
        using the backpropagation algorithm.

        The gradient of cross-entropy loss combined with softmax at the
        output layer simplifies to (y_predicted - y_true), which is used
        as the initial delta.

        Args:
            X:               input batch, shape (batch_size, n_features).
            y_onehot:        one-hot targets, shape (batch_size, n_classes).
            activations:     layer activations from the forward pass.
            pre_activations: pre-activation values from the forward pass.

        Returns:
            grad_w: list of weight gradient arrays, one per layer.
            grad_b: list of bias gradient arrays, one per layer.
        """
        batch_size = X.shape[0]
        n_layers = len(self.weights)
        grad_w = [None] * n_layers
        grad_b = [None] * n_layers

        # Output layer delta: d(CE)/d(z_out) = y_hat - y  (softmax + CE)
        delta = activations[-1] - y_onehot

        # Propagate gradients backwards through each layer
        for i in reversed(range(n_layers)):
            # Input to this layer
            layer_input = X if i == 0 else activations[i - 1]

            # Weight gradient: (input^T @ delta) / batch + L2 penalty
            grad_w[i] = (layer_input.T @ delta) / batch_size \
                        + self.l2_lambda * self.weights[i]

            # Bias gradient: mean of deltas across the batch
            grad_b[i] = np.mean(delta, axis=0, keepdims=True)

            # Propagate delta to the previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) \
                        * self._relu_derivative(pre_activations[i - 1])

        return grad_w, grad_b

    def _adam_update(self, grad_w, grad_b, m_w, v_w, m_b, v_b, t):
        """
        Update all weights and biases using the Adam optimiser.

        Adam maintains exponentially decaying running averages of the
        gradient (first moment, m) and the squared gradient (second
        moment, v). Bias correction compensates for their zero
        initialisation during early training steps.

        Args:
            grad_w, grad_b: gradients for weights and biases.
            m_w, v_w:       first/second moment estimates for weights.
            m_b, v_b:       first/second moment estimates for biases.
            t:              current timestep (epoch), for bias correction.
        """
        for i in range(len(self.weights)):
            # First moment update
            m_w[i] = self.beta1 * m_w[i] + (1 - self.beta1) * grad_w[i]
            m_b[i] = self.beta1 * m_b[i] + (1 - self.beta1) * grad_b[i]

            # Second moment update
            v_w[i] = self.beta2 * v_w[i] + (1 - self.beta2) * (grad_w[i] ** 2)
            v_b[i] = self.beta2 * v_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)

            # Bias-corrected estimates
            m_w_hat = m_w[i] / (1 - self.beta1 ** t)
            m_b_hat = m_b[i] / (1 - self.beta1 ** t)
            v_w_hat = v_w[i] / (1 - self.beta2 ** t)
            v_b_hat = v_b[i] / (1 - self.beta2 ** t)

            # Apply parameter updates
            self.weights[i] -= self.learning_rate * m_w_hat \
                               / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat \
                              / (np.sqrt(v_b_hat) + self.epsilon)

    @staticmethod
    def _relu(z):
        """
        ReLU activation: f(z) = max(0, z).

        Introduces non-linearity while being computationally cheap and
        avoiding the vanishing gradient problem of sigmoid/tanh.
        """
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        """
        Derivative of ReLU: 1 where z > 0, 0 otherwise.

        Used during backpropagation to gate the gradient flow through
        hidden layers.
        """
        return (z > 0).astype(np.float64)

    @staticmethod
    def _softmax(z):
        """
        Softmax activation for the output layer.

        Converts raw logits into a probability distribution over the
        four action classes. The max is subtracted for numerical
        stability.

        Args:
            z: pre-activation values, shape (batch_size, n_classes).

        Returns:
            Probability array of the same shape; each row sums to 1.
        """
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_pred, y_true):
        """
        Compute mean cross-entropy loss with L2 regularisation.

        Args:
            y_pred: predicted probabilities, shape (n_samples, n_classes).
            y_true: one-hot true labels, same shape.

        Returns:
            Scalar loss value (float).
        """
        n = y_pred.shape[0]
        clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        ce = -np.sum(y_true * np.log(clipped)) / n
        l2 = 0.5 * self.l2_lambda * sum(np.sum(w ** 2) for w in self.weights)
        return ce + l2

    @staticmethod
    def _one_hot(y, n_classes):
        """
        One-hot encode an array of integer class labels.

        Args:
            y:         1D numpy array of integer labels.
            n_classes: total number of distinct classes.

        Returns:
            2D array of shape (len(y), n_classes) with 1.0 at each label index.
        """
        encoded = np.zeros((y.shape[0], n_classes))
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    @staticmethod
    def _train_val_split(X, y, val_fraction):
        """
        Randomly split data into training and validation subsets.

        Args:
            X:            feature array, shape (n_samples, n_features).
            y:            target array (one-hot), shape (n_samples, n_classes).
            val_fraction: proportion of data reserved for validation.

        Returns:
            (X_train, y_train, X_val, y_val) tuple of numpy arrays.
        """
        n = X.shape[0]
        n_val = max(1, int(n * val_fraction))
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _legal_to_ints(self, legal):
        """
        Convert a list of legal actions into a set of integer labels.

        Handles both raw integer labels and Direction string objects
        to ensure compatibility with the Pacman framework's action representation.

        Args:
            legal: list of legal actions, or None.

        Returns:
            A set of integer action labels {0..3}, or empty set if None.
        """
        if legal is None:
            return set()

        result = set()
        for action in legal:
            if isinstance(action, int):
                result.add(action)
            elif hasattr(action, '__str__'):
                # Convert Direction strings (or similar objects) to ints
                name = str(action)
                if name in self._direction_to_int:
                    result.add(self._direction_to_int[name])
        return result

    def _safe_random_legal(self, legal):
        """
        Picks a random legal move. Used as a fallback when the model
        is untrained or all else fails.

        Args:
            legal: list of legal actions, or None.

        Returns:
            An integer in {0, 1, 2, 3}.
        """
        legal_ints = self._legal_to_ints(legal)
        if legal_ints:
            return int(np.random.choice(list(legal_ints)))
        return 0

# References:
# [1] ‘CS 188 - Pac-Man Project Information’, CS 188 Fall 2024.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://inst.eecs.berkeley.edu/~cs188/fa24/projects/


# Bibliography:
# [1] L. Hub, ‘Cross Entropy Loss function: A Simple Explanation for Everyone’, Medium.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://medium.com/@libertihub/cross-entropy-loss-function-a-simple-explanation-for-everyone-58a27f85f69b
# [2] ‘Early Stopping’, DeepAI.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://deepai.org/machine-learning-glossary-and-terms/early-stopping-machine-learning
# [3] J. Brownlee, ‘Gentle Introduction to the Adam Optimization Algorithm for Deep Learning’, MachineLearningMastery.com.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# [4] ‘Kaiming/He Initialization Explained’.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization
# [5] A. O. Hidayathullah, ‘Machine Learning: Understanding Perceptron’, Medium.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://medium.com/@AlvinOctaH/machine-learning-understanding-perceptron-1dcdfc78412a
# [6] ‘Neural networks | Machine Learning’, Google for Developers.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://developers.google.com/machine-learning/crash-course/neural-networks
# [7] ‘Overfitting: L2 regularization | Machine Learning’, Google for Developers.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://developers.google.com/machine-learning/crash-course/overfitting/regularization
# [8] ‘ReLU Activation Function in Deep Learning’, GeeksforGeeks.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
# [9] ‘Softmax Activation Function in Neural Networks’, GeeksforGeeks.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
# [10] P. Belagatti, ‘Understanding the Softmax Activation Function: A Comprehensive Guide’, SingleStore.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/
# [11] ‘What is Backpropagation? | IBM’.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.ibm.com/think/topics/backpropagation
# [12] ‘What is Perceptron’, GeeksforGeeks.
#       Accessed: Feb. 18, 2026. [Online].
#       Available: https://www.geeksforgeeks.org/machine-learning/what-is-perceptron-the-simplest-artificial-neural-network/