import numpy as np
import os
import gzip
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from collections import Counter

plt.rcParams.update({'font.size': 18})

# class names - labels corresponding to clothing items
lab = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# helper function. plots the confusion matrix based on classification results
def plot_cm(labels, res):
    cm = confusion_matrix(labels, res)
    
    # Here the cm is visualized using sns 
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, cmap = "BuPu", xticklabels = lab, yticklabels = lab)
    plt.title('Confusion matrix on Fashion MNIST')
    plt.show()

## util function provided by Zalando research
## source: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
# used to load the dataset with the given train test split
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

# main logic and modeling section
def main():
    # data loading
    print('Loading data..')
    X_train, y_train = load_mnist('fashion_mnist/data/fashion', kind='train')
    X_test, y_test = load_mnist('fashion_mnist/data/fashion', kind='t10k')

    # kernels and c values for which we experiment
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    c_vals = [1.0, 10.0]

    # iterate over all possible kernel - c value pairs
    for k in kernels:
        for c in c_vals:
            print('Current model: {} kernel and c = {}'.format(k, c))

            # initialize and train model with current parameter set
            model = SVC(C=c, kernel=k, verbose=True)
            print('Training..')
            model.fit(X_train, y_train)

            # test on the respective set
            print('Predicting..')
            res = model.predict(X_test)

            # evaluation metrics - cm, f score, precision, recall, accuracy
            print('Evaluating results')
            print(classification_report(y_test, res))
            plot_cm(y_test, res)

if __name__ == "__main__":
    main()