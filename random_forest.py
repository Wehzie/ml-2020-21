import numpy as np
import os
import gzip
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler 
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

# clothing label names
labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

## util function provided by Zalando research
## source: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


## util function for plotting the confusion matrix 
def plot_cm(ground_truth, results, rf_name):   
    cm = confusion_matrix(ground_truth, results)
    
    # visualize confusion matrix
    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(cm, cmap = "BuPu", xticklabels = labels, yticklabels = labels)
    ax.set_title('Confusion matrix for Random Forest Classifier on Fashion MNIST')
    #plt.show()
    plt.savefig(rf_name)


## setup of the random forest classifier
def rf_classif(estimators, crit, depth):
    classifier_name = str(estimators) + crit + str(depth)
    return RandomForestClassifier(n_estimators = estimators, criterion = crit, max_depth = depth, random_state = 0), classifier_name

print('Loading data..')
X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')
print('Data ready')

## choose classifier params (N tree estimators, impurity criterion 'gini' or 'entropy', max_depth of tree estimators in forest)
rf, rf_name = rf_classif(150, 'entropy', None)
print('Fitting classifier')
rf.fit(X_train, y_train)

estimators = [(est.get_depth(), est.tree_.max_depth, est.max_depth) for est in rf.estimators_]
max_estimators = np.max([depth for (depth, _, _) in estimators])
print('Max depth out of all estimators: ', max_estimators)

print('Predicting')
preds = rf.predict(X_test)

## evaluation
print(classification_report(y_test, preds, target_names = labels))

## confusion matrix
plot_cm(y_test, preds, rf_name)
