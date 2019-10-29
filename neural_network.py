import numpy as np
import random

import matplotlib.pyplot as plt


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    soft_max = exp_z / exp_z.sum(axis=0)
    return soft_max

def weight_initialization(x_shape, nndim, y_shape):
    W1 = np.random.randn(x_shape, nndim)
    b1 = np.zeros((1,nndim))
    W2 = np.random.randn(nndim,y_shape)
    b2 = np.zeros((1,y_shape))
    parameters = {
        "W1": W1,
        "b1" : b1,
        "W2": W2,
        "b2" : b2
      }
    return parameters

def feedfoward(X,W1,W2,b1,b2):
        a = np.dot(X,W1) + b1
        #print(a.shape)
        h = np.tanh(a)
        #print(h.shape)
        z = np.dot(h,W2) + b2
        #print(z.shape)
        y_pred = softmax(z)
        #print(y_pred.shape)
        return a,h,z,y_pred

def calculate_loss(model,X, y):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    a,h,z,y_pred = feedfoward(X,W1,W2,b1,b2)
    if y_pred==0:
        logY = np.multiply(y, 0)
        logY_= np.multiplty(1-y,np.log2(1-y_pred))
        loss = -np.sum(logY + logY_)/2 
    elif y_pred==1:
        logY = np.multiply(y, np.log2(y_pred))
        logY_ = np.multiply(1-y,0)
        loss = -np.sum(logY + logY_)/2 
    else:
        loss = -np.sum(np.multiply(y, np.log(y_pred)) +  np.multiply(1-y, np.log(1-y_pred)))/X.shape[0]
    
    #print (loss)
    #cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2)))/m
    loss = np.squeeze(loss)
    cost = {
    "a": a,
    "h": h,
    "z": z,
    "y_pred": y_pred,
    "loss":loss
    }
    return cost

def backward_prop(X, Y, cost, parameters):
    a = cost['a']
    h = cost['h'] 
    y_pred = cost['y_pred']
    W2 = parameters['W2']
    X = np.reshape(X,(1,2))
    dZ2 = np.subtract(y_pred,Y)
    dW2 = np.dot(h.T,dZ2)
    db2 = dZ2#np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(dZ2,W2.T), 1-np.power(np.tanh(a),2))
    dW1 = np.dot(X.T,dZ1)
    db1 = dZ1#np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
    }

    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
    }

    return new_parameters

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    c= 0
    learning_rate = 0.1
    parameters= weight_initialization(X.shape[1], nn_hdim, 1)
    for i in range(num_passes):
        if i % X.shape[0]-1==0:
            c = 0
        cost = calculate_loss(parameters,X[c],y[c])
        if print_loss == True:
            print(cost['loss'])
        grads = backward_prop(X[c], y[c], cost, parameters)
        parameters= update_parameters(parameters, grads, learning_rate)
        print(c)
        print(i)
        c = c +1
    return parameters

## NOT SURE ABOUT THIS FUNCTION THOUGH
def predict(model,X):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    a,h,z,y_pred = feedfoward(X,W1,W2,b1,b2)
    return y_pred
        
def plot_decision_boundary(pred_func, X, y ) :
    # Set min and max values  and  give  i t  some padding
    xmin,xmax = X[:,0].min()-.5,  X[:,0].max()+.5
    ymin, ymax = X[:,1].min()-.5,  X[:,1].max()+.5
    h  =  0.01
    # Generate a  grid  of  points  with  distance  h between them
    xx, yy  =  np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
    # Predict  the  function  value  for  the  whole  gid
    Z  =  pred_func(np.c_[xx.ravel(),yy.ravel()])
    Z  =  Z.reshape(xx.shape)
    # Plot  the  contour and  training  examples
    plt.contourf( xx , yy,Z , cmap=plt.cm.Spectral)
    plt.scatter(X[:,0 ],X [:,1],c=y,cmap=plt.cm.Spectral)