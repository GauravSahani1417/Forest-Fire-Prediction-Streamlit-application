# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:47:29 2020

@author: gaurav sahani
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('Forest_fire.csv.txt')
df.head()

df.drop(['Area'],axis=1,inplace=True)

from sklearn.tree import DecisionTreeClassifier

X=df[['Oxygen','Temperature','Humidity']]
y=df[['Fire Occurrence']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

pickle.dump(dt,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))