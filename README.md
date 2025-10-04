# Deep Neural Network from Scratch (Cat vs Non-Cat Classification)

## Overview
This project implements a **fully connected L-layer neural network** from scratch in Python using **NumPy**, without high-level frameworks like TensorFlow or PyTorch.  
The model is designed to classify images of cats vs non-cats using a logistic regression-inspired architecture with **ReLU activations** in hidden layers and **Sigmoid activation** in the output layer.

---

## Neural Network Architecture

The network follows this structure:

- **Input layer:** `12288` units (64×64×3 images flattened)
- **Hidden layer 1:** `20` units, ReLU activation
- **Hidden layer 2:** `7` units, ReLU activation
- **Hidden layer 3:** `5` units, ReLU activation
- **Output layer:** `1` unit, Sigmoid activation

**Activation functions:**

- ReLU for hidden layers → `A = max(0, Z)`
- Sigmoid for output → `A = 1 / (1 + exp(-Z))`

**Weight Initialization:** `He` initialization for ReLU layers.


