# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:44:29 2020

@author: MBedouret
"""

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

plt.plot(forest.feature_importances_, 'o') 
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90);