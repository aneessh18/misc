import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def lms(y_hat, y):
    n, m = y.shape
    return np.sum(np.square(y_hat-y))/(2)

def get_digit_recogniton_data():
    digit_data = pd.read_csv('train.csv')
    labels = digit_data['label']
    images = digit_data.drop(['label'], axis=1)
    m = len(images)
    images, labels = np.array(images).T, np.array(labels)
    images = images/255
    one_hot_labels = np.zeros((10,m))
    for i in range(m):
        one_hot_labels[labels[i]][i] = 1
    return images, one_hot_labels


class Model:
    def __init__(self, architecture):
        self.parameters = self.initialize_parameters(architecture) #initialize the weights
        self.architecture = architecture
        
    def initialize_parameters(self, layer_dims):

        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1])
            parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
        return parameters
    
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z
    
    def forward(self, X, parameters):
    
        A = X
        L = len(parameters)
        L //=2
        weighted_sums = {}
        activations = {}
        activations[0] = X
        for l in range(1, L+1):
            weighted_sums[l] = self.linear_forward(A, parameters["W"+str(l)], parameters['b'+str(l)])
            activations[l] = sigmoid(weighted_sums[l])
            A = activations[l]

        return weighted_sums, activations
    
    def update_parameters(self, parameters, grads, learning_rate, previous_updates, momentum):
        L = len(parameters) // 2 

        for l in range(L):
            present_update_w = previous_updates['W'+str(l+1)]*momentum + (1-momentum)*grads['dW'+str(l+1)]
            present_update_b = previous_updates['b'+str(l+1)]*momentum + (1-momentum)*grads['db'+str(l+1)]
            parameters["W" + str(l+1)] = parameters['W'+str(l+1)] - learning_rate * present_update_w
            parameters["b" + str(l+1)] = parameters['b'+str(l+1)] - learning_rate * present_update_b
            previous_updates['W'+str(l+1)] = present_update_w
            previous_updates['b'+str(l+1)] = present_update_b


        return parameters
    
    def backward(self, weighted_sums, activations, y, parameters):
    
        L = len(activations)-1 #no of layers
        dA_prev = 0 #random init
        grads = {}
        m = y.shape[1]
        for l in range(L, 0, -1):
            if l == L: #last layer 
                dA_prev = activations[l]*(1-activations[l])*(activations[l]-y)
            else:
                dA_prev = np.dot(parameters['W'+str(l+1)].T, dA_prev)*(activations[l])*(1-activations[l])

            grads['dW'+str(l)] = (np.dot(dA_prev, activations[l-1].T))/m
            grads['db'+str(l)] = (np.sum(dA_prev, axis=1, keepdims=True))/m

        return grads
    
    def train(self,X, Y,learning_rate = 0.01, epochs = 15, batch_size=10, momentum=0.98):
    
    
        epoch_costs = []
        parameters = self.parameters
        L = len(self.architecture)-1 #no of layers
        m = X.shape[1]
        batch_epochs = m//batch_size
        previous_updates = defaultdict(int)
        for i in range(epochs):
            epoch_cost = 0
            for j in range(batch_epochs):
                mini_batch_X = X[:,j*batch_size:(j+1)*batch_size]
                mini_batch_Y = Y[:,j*batch_size:(j+1)*batch_size]
                weighted_sums, activations = self.forward(mini_batch_X, parameters)
                epoch_cost += lms(activations[L], mini_batch_Y)
                grads = self.backward(weighted_sums, activations, mini_batch_Y, parameters)
                parameters = self.update_parameters(parameters, grads, learning_rate,previous_updates, momentum=momentum)

            epoch_costs.append(epoch_cost)
        
        plt.plot(np.squeeze(epoch_costs))
        plt.ylabel('epoch cost')
        plt.xlabel('no of epochs')
        plt.show()

        return parameters
    
def predict(valid_images, valid_labels, parameters):
    
    weighted_sum, activations = sigmoid_model.forward(valid_images, parameters)
    L = len(parameters)//2
    AL = activations[L]
    n, m = AL.shape
    correct = 0
    
    for i in range(m):
        index = np.argmax(AL[:,i])
        j = 0
        for k in range(n):
            if valid_labels[k][i] == 1:
                j = k
                
        correct += 1 if index == j else 0
    
    print("Model accuracy on validation data", (correct/m))

if __name__ == "__main__":

    digit_images, digit_labels = get_digit_recogniton_data()    
    train_images = digit_images[:,:10000]
    train_labels = digit_labels[:,:10000]
    sigmoid_model = Model([784,128,10])
    weights_sigmoid = sigmoid_model.train(X = train_images, Y = train_labels, epochs=10, learning_rate=0.1, batch_size=10, momentum=0.9)
    
    valid_images = digit_images[:,10000:30000]
    valid_labels = digit_labels[:,10000:30000]
    predict(valid_images, valid_labels, weights_sigmoid)