# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""
Predicitve_Analytics.py
"""


def Accuracy(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    a = ConfusionMatrix(y_true, y_pred)
    total_correct = 0
    total = len(a)

    for i in range(0, total):
            total_correct += a[i][i]
    acc = total_correct / total

    return acc


def Recall(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    a = ConfusionMatrix(y_true, y_pred)
    total = len(a)
    tp = 0
    fn = 0
    for i in range(0, total):
        tp += a[i][i]
        fn = fn + np.sum(a.transpose()[i][:i]) + np.sum(a.transpose()[i][i + 1:])
    return tp / (tp + fn)


def Precision(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    a = ConfusionMatrix(y_true, y_pred)

    total = len(a)
    tp = 0
    fp = 0
    for i in range(0, total):
        tp += a[i][i]
        fp = fp + np.sum(a[i][:i]) + np.sum(a[i][i + 1:])
    return tp / (tp + fp)

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """


def ConfusionMatrix(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    #Method for confusion matrix computation discussed in class 3/2/2020
    classCount =  len(np.histogram(y_true)[1])
    step_1 = (y_true * classCount) + y_pred
    hist = np.histogram(step_1, bins=classCount**2)
    return hist[0]


def cosine_dist(X_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    """
    return np.dot(X_train/np.linalg.norm(X_train), X_test.T/np.linalg.norm(X_test))

def KNN(X_train, X_test, Y_train, K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """
    neighbours = []
    for i in range(len(X_test)):
        point_dist = cosine_dist(X_train, X_test[i])  # Compute the cos dist between the entire dataset(X_train) and the testpoint(X_train[i])
        dist = np.concatenate([point_dist, Y_train]).reshape(2, len(Y_train)).T  # Combine the distances(point_dist) with the labels(Y_train), reshape and tranpose
        lst = dist.tolist()  # Convert to python list to allow for key based sorting
        lst.sort(key=lambda x: x[0])
        sorted_KNN = np.asarray(lst).T[1].tolist()  # Convert to numpy array to transpose, take the row containing the distances and convert it back to python list
        neighbours.append(max(set(sorted_KNN[0:K]),key=sorted_KNN.count))  # Get the mode value from the list of all points and base the class label based on that
    return np.asarray(neighbours)


def RandomForest(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """


def PCA(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    means = np.mean(X_train.transpose(),axis = 1)

    xku = np.subtract(X_train, means)
    cov = np.divide(np.dot(xku.transpose(), xku),(X_train.shape[0]-1))

    U, Sigma, Vh = np.linalg.svd(cov,full_matrix=False,compute_uv=False)
    P = np.dot(U,X_train)

    return P

def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """


def SklearnSupervisedLearning(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: List[numpy.ndarray]
    """
    sclass = SVC(kernel='linear')
    sclass.fit(X_train, Y_train)
    predsvm = sclass.predict(X_test)

    logReg = LogisticRegression()
    logReg.fit(X_train, Y_train)
    predlogreg = logReg.predict(X_test)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    preddt = dt.predict(X_test)

    kn = KNeighborsClassifier(n_neighbors=10)
    kn.fit(X_train, Y_train)
    predkn = kn.predict(X_test)

    ret = [predsvm, predlogreg, preddt, predkn]

    # what would the parameters be for the accuracy_score below

    return ret


def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """

#Create dataset with sklearn
def getDataset():

    dataset = pd.read_csv("./data.csv").to_numpy()

    y = []
    X = []
    for i in dataset:
        y.append(i[len(i) - 1]) #takes the  very last value and puts it into the vector y
        X.append(i[:-1]) # gets rid of the last column because the last column represents the group number or y value
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

#Main function for testing purposes
def main():
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :type Y_test: numpy.ndarray

    """
    X_train, X_test, Y_train, Y_test = getDataset()
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Y_test = np.asarray(Y_test, dtype=np.float32)
    MatplotVisualize(X_train, Y_train, X_test, Y_test)
"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""

def gridSearch(x_train, y_train):
    logreg_params = [
        {'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000]}
    ]
    svc_params = [
        # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10], 'kernel': ['rbf'], 'gamma': [0.1, 0.5, 0.9]}
    ]
    knn_params = [
        {'n_neighbors': [5, 6, 7, 8, 9]}
    ]
    dt_params = [
        {'criterion': ["gini", "entropy"], 'splitter': ["best", "random"]}
    ]

    log_reg = LogisticRegression()
    sv_class = SVC(kernel='rbf')
    knn_class = KNeighborsClassifier(n_neighbors=5)
    dt_class = DecisionTreeClassifier()

    classifier = [sv_class, log_reg, knn_class, dt_class]
    params = [svc_params, logreg_params, knn_params, dt_params]
    type = ["SVM", "logistic regression", "KNN", "decision tree"]

    ret_val = []

    for i in range(len(type)):
        grid_search = ms.GridSearchCV(estimator=classifier[i],
                                      param_grid=params[i],
                                      scoring='accuracy',
                                      n_jobs=-1
                                      )
        grid_search.fit(x_train, y_train)
        print("For", type[i], ":")
        accuracy = grid_search.best_score_
        print("Best Accuracy Score:", accuracy)
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        ret_val.append(best_params)
    return best_params


def MatplotSklearn(X_train, Y_train):
    """
    :type X_train: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: List[numpy.ndarray]
    """
    sclass = SVC(kernel='linear')
    svmClass = sclass.fit(X_train, Y_train)

    logReg = LogisticRegression()
    logRegClass = logReg.fit(X_train, Y_train)

    dt = DecisionTreeClassifier()
    dtClass = dt.fit(X_train, Y_train)

    knn = KNeighborsClassifier(n_neighbors=10)
    knnClass = knn.fit(X_train, Y_train)

    data = [svmClass, logRegClass, dtClass, knnClass]

    return data

def MatplotVisualize(X_train, Y_train, X_test, Y_test):
    pred_data = MatplotSklearn(X_train, Y_train)
    titles = ["SVM", "Logistic Regression", "Decision Tree", "K Nearest Neighbors"]
    for i in range(len(pred_data)):
        cm = metrics.plot_confusion_matrix(pred_data[i], X_test, Y_test)
        cm.ax_.set_title(titles[i])
    plt.show()

main()
