# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:14:57 2020

@author: MBedouret
"""

from sklearn.tree import DecisionTreeClassifier
from  sklearn import model_selection
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = model_selection.train_test_split( cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("accuracy on training set: %f" % tree.score(X_train, y_train)) 
print("accuracy on test set: %f" % tree.score(X_test, y_test))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("------------")
print("accuracy on training set: %f" % tree.score(X_train, y_train)) 
print("accuracy on test set: %f" % tree.score(X_test, y_test))
print("Feature importances:")

print(tree.feature_importances_)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
