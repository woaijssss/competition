import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_bacward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ


# layer_dim is the layer parameters of the network
# such as [5,4,3] means input layer 5,hidden layer 4, output layer 3
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_act_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, act_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, act_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, act_cache)
    return A, cache


# X:data input size is (input size, number of example)
# parameters:output of initialize_parameters
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_act_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_act_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    assert (AL.shape == (1, X.shape[1]))
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    Eri = Y - AL
    minus_Eri = Eri[Eri <= 0]
    minus = np.power(2, minus_Eri/5)
    minus_sum = np.sum(minus)
    pos_Eri = Eri[Eri > 0]
    pos = np.power(0.5, pos_Eri/20)
    pos_sum = np.sum(pos)
    cost = (pos_sum + minus_sum)/m
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_act_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_bacward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = np.zeros((1, m))
    Eri = AL-Y
    for i in range(0, AL.shape[1]):
        if Eri[0][i] <= 0:
            dAL[0][i] = np.power(2, Eri[0][i]/5) * np.log(2) * (1/5)
        else:
            dAL[0][i] = np.power(0.5, Eri[0][i]/20) * np.log(2) * (-1/20)
    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = linear_act_backward(dAL, current_cache,
                                                                                       'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_act_backward(grads['dA'+str(l+1)], current_cache, 'relu')
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]

    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate = 0.01, num_iterations = 3000):
    costs = []
    parameters = initialize_parameters(layer_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            print('cost after iteration %i:%f'%(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    data = pd.read_csv('final.csv')
    Y_data = np.array(data['last_time'])
    Y = Y_data.reshape((1, 303159))
    X_data = np.array(data.drop('last_time', axis=1))
    X = X_data.reshape((8, 303159))
    layer_dims = [8, 4, 1]
    parameters = L_layer_model(X, Y, layer_dims, 0.01, 3000)



