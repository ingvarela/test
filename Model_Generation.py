# -*- coding: utf-8 -*-
"""

@author: svs26
"""

#Dataset Source: https://www.kaggle.com/code/caesarmario/petal-profiling-classification-clustering/notebook#3.-%7C-Reading-Dataset-%F0%9F%91%93

import numpy as np
import pandas as pd

df=pd.read_csv('iris//iris.csv')

df



#Separate Inputs(Features) From Output(Classes)
X = df.iloc[1:,:-1]
y = df.iloc[1:,-1]

X.head()

# replacing values
y.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                        [0, 1, 2], inplace=True)

y.head()




### Train Test Split
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

### Implement Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train.values,y_train.values)

### Prediction
y_pred=classifier.predict(X_test.values)


### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


### Create a Pickle file using serialization
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

prediction = classifier.predict([[6.4,3.2,5.3,2.3]])
