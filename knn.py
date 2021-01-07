
'''
sciki-learn KNN-classifier applied to MNIST-fashion dataset.

sciki-learn KNN-documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    https://scikit-learn.org/stable/modules/neighbors.html#classification
'''

# import data from OpenML on openml.org
#from sklearn.datasets import fetch_openml
#X_train, y_train = fetch_openml("Fashion-MNIST", return_X_y=True, as_frame=False)

from fashion_mnist.utils import mnist_reader

# X holds data, y holds labels
X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# classify test data, return list of labels
#classifier.predict(X_test)

# classify test data, for each test image return a list with the probability of belonging to each class
#classifier.predict_proba(X_test)

# classify test data, compare against test labels, return accuracy
#print(classifier.score(X_test, y_test))

from sklearn.metrics import classification_report

target_names = ["0 t-shirt", "1 trousers", "2 pullover", "3 dress", "4 coat", "5 sandal", "6 shirt", "7 sneaker", "8 bag", "9 boot"]

print(classification_report(y_test, classifier.predict(X_test), target_names))
