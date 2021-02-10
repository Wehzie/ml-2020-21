import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from fashion_mnist.utils import mnist_reader

'''
sciki-learn KNN-classifier applied to MNIST-fashion dataset.

sciki-learn KNN-documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    https://scikit-learn.org/stable/modules/neighbors.html#classification
'''

class Knn:
'''
Define methods to run KNN.
Load data and store results.
'''

    def __init__(self):
        self.load_data()
        self.results = pd.DataFrame(columns=['k', 'weight', 'accuracy'])
    
    # load MNIST data
    def load_data(self):
        # X holds data, y holds labels
        self.X_train, self.y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
        self.X_test, self.y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

    # initialize and train classifier object on training data
    def train_classifier(self, k, weight):
        self.classifier = KNeighborsClassifier(n_neighbors=k, weights=weight)
        self.classifier.fit(self.X_train, self.y_train)

    # test classifier on testing data given a set of parameters and store results
    def test_classifier(self, k, weight):
        entry = {
            'k': k,
            'weight': weight,
            'accuracy': self.classifier.score(self.X_test, self.y_test)
        }
        self.results = self.results.append(entry, ignore_index=True)
        print(f"Accuracy: {entry['accuracy']}")

    # train and test classifier using over a range of parameters
    def test_parameters(self):
        ks = [1, 5]
        weights = ['uniform', 'distance']
        #algorithms = ['brute', 'kd_tree', 'ball_tree']
        
        for k in ks:
            for weight in weights:
                print(f"\nTesting k: {k}, weight: {weight}")
                self.train_classifier(k, weight)
                self.test_classifier(k, weight)

    # print all test results and highlight best parameters
    def best_parameters(self):
        print(self.results)
        best_row = self.results['accuracy'].argmax()    
        print(f"\nBest parameters:\n{self.results.loc[[best_row]]}")

# intialize a KNN object and run its methods.
def main():
    knn = Knn()
    knn.test_parameters()
    knn.best_parameters()

main()
