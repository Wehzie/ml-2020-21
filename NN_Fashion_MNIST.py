# README: Make sure to read all comments before running. 
# No graphical interface is given and depending on the goal of running 
# some things need to be "hacked"/copy-pasted/uncommented into or out of the code.

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust path to where project is located files are located.
sys.path.insert(1, '/home/paul/Documents/python projects/')

from Machine_Learning_Project.fashion_mnist_master.utils import mnist_reader

import tensorflow as tf
from tensorflow import keras

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Adjust path to where fashion_mnist files are located.
dataPath = 'Documents/python projects/Machine_Learning_Project/fashion_mnist_master/data/fashion'

# Getting data and transforming
X_train, y_train = mnist_reader.load_mnist(dataPath, kind='train')
X_test, y_test = mnist_reader.load_mnist(dataPath, kind='t10k')
train_images = X_train/255.0
test_images = X_test/255.0

# Function to build a keras model given hyperparameters
# Number of layers muyst be added here manually
def build_model(a_function, layer_size, bias):
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(layer_size, activation=a_function, bias_initializer=keras.initializers.Constant(bias/10)),
            keras.layers.Dense(layer_size, activation=a_function, bias_initializer=keras.initializers.Constant(bias/10)),
            keras.layers.Dense(10, activation="softmax")
            ])
    return model

# function to plot as confusion matrix
def plot_cm(cm):
    labels = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    # Magic bit of code credited to Alina
    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(cm, cmap = "BuPu", xticklabels = labels, yticklabels = labels)
    ax.set_title('Confusion matrix for MLP classifier on Fashion MNIST', size=18)

    # Rotate axis
    plt.xticks(rotation=45, size=15)
    plt.yticks(rotation=45, size=15)
    plt.show()


# Initializing lists
acc_list = []
precision_list = []
recall_list = []
F1_list = []
loss_list = []

max_list = []
min_list = []

# Given activation function(s) uncomment as needed
# function_list = ["relu" , "sigmoid", "tanh"]
function_list = ["relu"]

# Parameter sweep activation function loop
# Loops can be set to only test a certain model
for a_function in function_list:

    # Initialize min and max model scores
    max_model_score = 0
    min_model_score = 1

    # Parameter sweep epoch loop
    for episode in range(10,11):
        
        # Parameter sweep layer size loop
        for layer_size in range(170,190,20):

            # Parameter sweep bias loop
            for bias in range(6, 7, 2):

                # Building and fitting model
                model = build_model(a_function,layer_size,bias)
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(train_images, y_train, epochs=episode)

                # Getting evaluation measures
                test_loss, test_acc = model.evaluate(test_images, y_test)

                yhat_probs = model.predict(test_images, verbose=0)
                yhat_classes = np.argmax(model.predict(test_images, verbose=0), axis=-1)

                acc_list.append(test_acc)
                print("Tested Acc: ", test_acc)

                test_prec = precision_score(y_test, yhat_classes, average="macro")
                precision_list.append(test_prec)
                print("Tested Precision: ", test_prec)

                test_recall = recall_score(y_test, yhat_classes, average="macro")
                recall_list.append(test_recall)
                print("Tested Recall: ", test_recall)

                test_F1 = f1_score(y_test, yhat_classes, average="macro")
                F1_list.append(test_F1)
                print("Tested F1: ", test_F1)

                loss_list.append(test_loss)

                # Finding the best model given mean of different measures
                total_model_score = (test_acc + test_prec + test_recall + test_F1)/4
                if total_model_score > max_model_score:
                    max_model_score = total_model_score
                    max_bias = bias
                    max_layer = layer_size
                    max_episode = episode
                    max_function = a_function

                if total_model_score < min_model_score:
                    min_model_score = total_model_score
                    min_bias = bias
                    min_layer = layer_size
                    min_episode = episode
                    min_function = a_function
    
    print("maxi: ", max_model_score, max_bias, max_layer, max_episode, max_function)
    print("mini: ", min_model_score, min_bias, min_layer, min_episode, min_function)

    # Saving best models to print
    max_list.append(max_model_score)
    max_list.append(max_bias)
    max_list.append(max_layer)
    max_list.append(max_episode)
    max_list.append(max_function)
    
    min_list.append(min_model_score)
    min_list.append(min_bias)
    min_list.append(min_layer)
    min_list.append(min_episode)
    min_list.append(min_function)

# Print best models
print("max: ", max_list)
print("min: ", min_list)

# Make confusion matrix of last model
cm = sklearn.metrics.confusion_matrix(yhat_classes, y_test)
plot_cm(cm)



