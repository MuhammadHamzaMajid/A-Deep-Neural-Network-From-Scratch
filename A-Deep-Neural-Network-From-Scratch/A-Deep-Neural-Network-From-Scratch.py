import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

#L-layer neural network architecture with final layer having a sigmoid activation, and the others having a relu activation

#Define helper sigmoid and relu functions
def sigmoid(Z):
    A = 1 / (1+np.exp(-Z))
    activation_cache = Z
    return A, activation_cache

def relu(Z):
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy = True)#just to keep the same shape
    dZ[Z <= 0] = 0
    return dZ

#1. Initialize the parameters
#a. Initialize parameters for an L-layer network 
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) #no. of layers
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1]) #He initialization
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))
    return parameters

#2. Forward Propagation Module
#a. linear forward
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

#b. linear forward activation
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

#c. L-layer Model's Forward Propagation
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    #implement linear -> relu (loop starts from layer1 because layer0 = input)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation="relu")
        caches.append(cache)
    
    #implement linear -> sigmoid for the final layer
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

#3. Compute the Cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)) / m
    cost = np.squeeze(cost)
    return cost

#4. Backward Propagation Module
#linear-backward -> linear-activation-backward -> L-layer-backward

#a. linear_backward
def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

#b. linear_activation_backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

#c. L-layer backward propagation
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    #initialize back propagation
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    #Lth layer(sigmoid layer) grads
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA"+str(L-1)] = dA_prev_temp
    grads["dW"+str(L)] = dW_temp
    grads["db"+str(L)] = db_temp

    #loop from l = l-2 to l = 0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp

    return grads

#5. Update Parameters
def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 #no. of layers
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]
    return parameters

#6. Load the Dataset
train_x_orig = h5py.File('E:\\Deep-Learning-Specialization\\Logistic-Regression-from-Scratch\\train_catvsnoncat.h5', 'r')['train_set_x'][:]
test_x_orig = h5py.File('E:\\Deep-Learning-Specialization\\Logistic-Regression-from-Scratch\\test_catvsnoncat.h5', 'r')['test_set_x'][:]
train_y = h5py.File('E:\\Deep-Learning-Specialization\\Logistic-Regression-from-Scratch\\train_catvsnoncat.h5', 'r')['train_set_y'][:]
test_y = h5py.File('E:\\Deep-Learning-Specialization\\Logistic-Regression-from-Scratch\\test_catvsnoncat.h5', 'r')['test_set_y'][:]
classes = h5py.File('E:\\Deep-Learning-Specialization\\Logistic-Regression-from-Scratch\\train_catvsnoncat.h5', 'r')['list_classes'][:]

#7. Pre-process the Dataset
#a. reshape the dataset
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_y = train_y.reshape(1, -1)
test_y = test_y.reshape(1, -1)

#b. standardize
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

#8. Combining all the Functions and Building a Neural Network

#a. const layer_dims
layers_dims = [12288, 20, 7, 5, 1] #4-layer model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        #forward propagation
        AL, caches = L_model_forward(X, parameters)
        #compute cost
        cost = compute_cost(AL, Y)
        #backward propagation
        grads = L_model_backward(AL, Y, caches)
        #update parameters(gradient descent)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs

#9. Apply the neural network model on our dataset
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost=True)

#10. Get the Accuracies
def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    AL, caches = L_model_forward(X, parameters)

    # Convert probabilities AL to 0/1 predictions
    for i in range(AL.shape[1]):
        p[0, i] = 1 if AL[0, i] > 0.5 else 0

    # Print accuracy
    print("Accuracy: {} %".format(100 - np.mean(np.abs(p - Y)) * 100))

    return p

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


