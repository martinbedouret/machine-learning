# -*- coding: utf-8 -*-

#data 
from sklearn.datasets import load_iris
iris_dataset=load_iris()
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data' ],iris_dataset['target' ],random_state=0 )
print ( "Target names:" , iris_dataset [ 'target_names' ])

#plotting 
import pandas as pd
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names )
pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',alpha=0.8,s=60,hist_kwds={'bins': 20})

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit ( X_train , y_train )

#Making Predictions
import numpy as np
X_new = np . array ([[ 5 , 2.9 , 1 , 0.2 ]])
prediction = knn . predict ( X_new )
print ( "Predicted target name:" , iris_dataset [ 'target_names' ][ prediction ])

X_new = np . array ([[ 7,4.1,1,1]])
prediction = knn . predict ( X_new )
print ( "Predicted target name:" , iris_dataset [ 'target_names' ][ prediction ])

y_pred = knn . predict ( X_test ) 
print ( "Test set predictions: \n " , y_pred )

#Evaluating the Model
print ( "Test set score: {:.2f}" . format ( np . mean ( y_pred == y_test )))