# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:20:24 2018

@author: pulki
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
resource = pd.read_csv('resources.csv')
resource = resource.drop('description' , 1)
resource = resource.set_index('id').sum(level = 0).reset_index()
dataset = dataset.merge(resource , how= 'left' , on = 'id')
y = dataset.iloc[:,15]


dataset = dataset.iloc[: , 2:]
dataset = dataset.drop('project_is_approved' ,1)
dataset = dataset.drop('project_essay_1' ,1)
dataset = dataset.drop('project_essay_2',1)
dataset = dataset.drop('project_essay_3',1)
dataset = dataset.drop('project_essay_4',1)
dataset = dataset.drop('project_title',1)
dataset = dataset.drop('project_resource_summary',1)
year = list()
month = list()
for k in range(0,182080):
   year.append(dataset.iloc[k ,2].split('-')[0])
   month.append(dataset.iloc[k ,2].split('-')[1])
dataset['year'] = year   
dataset['month'] = month   
dataset = dataset.drop('project_submitted_datetime',1)
dataset['cost'] = dataset['quantity']*dataset['price']
dataset = dataset.drop('quantity',1)
dataset = dataset.drop('price',1)


#dealing with misssing data
dataset = dataset.fillna(dataset['teacher_prefix'].value_counts().index[0])
dataset = dataset.fillna(dataset['school_state'].value_counts().index[0])
dataset = dataset.fillna(dataset['project_grade_category'].value_counts().index[0])
dataset = dataset.fillna(dataset['project_subject_categories'].value_counts().index[0])
dataset = dataset.fillna(dataset['project_subject_subcategories'].value_counts().index[0])
dataset = dataset.fillna(dataset['teacher_number_of_previously_posted_projects'].value_counts().index[0])
dataset = dataset.fillna(dataset['year'].value_counts().index[0])
dataset = dataset.fillna(dataset['month'].value_counts().index[0])
dataset.iloc[:,2].fillna(dataset.iloc[: , 8].mean() , inplace = True)


from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
le_2 = LabelEncoder()
le_3 = LabelEncoder()
le_4 = LabelEncoder()
le_5 = LabelEncoder()
le_6 = LabelEncoder()
le_7 = LabelEncoder()
dataset.iloc[:,0] = le_1.fit_transform(dataset.iloc[:,0])
dataset.iloc[:,1] = le_2.fit_transform(dataset.iloc[:,1])
dataset.iloc[:,2] = le_3.fit_transform(dataset.iloc[:,2])
dataset.iloc[:,3] = le_4.fit_transform(dataset.iloc[:,3])
dataset.iloc[:,4] = le_5.fit_transform(dataset.iloc[:,4])
dataset.iloc[:,6] = le_6.fit_transform(dataset.iloc[:,6])
dataset.iloc[:,7] = le_7.fit_transform(dataset.iloc[:,7])

from sklearn.preprocessing import OneHotEncoder
onehotencoder_1 = OneHotEncoder(categorical_features = [0])
dataset = onehotencoder_1.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_2 = OneHotEncoder(categorical_features = [1])
dataset = onehotencoder_2.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_3 = OneHotEncoder(categorical_features = [2])
dataset = onehotencoder_3.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_4 = OneHotEncoder(categorical_features = [3])
dataset = onehotencoder_4.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_5 = OneHotEncoder(categorical_features = [4])
dataset = onehotencoder_5.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_6 = OneHotEncoder(categorical_features = [6])
dataset = onehotencoder_6.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]
onehotencoder_7 = OneHotEncoder(categorical_features = [7])
dataset = onehotencoder_7.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)



#Now  making the ANN

import keras
from keras.models import Sequential      # this is used to iinitialise pur neural network
from keras.layers import Dense        # this is used to make the different layers ofour nueral network

#initialising the ANN
classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense( units = 30 , input_shape = (61,) , kernel_initializer= 'uniform' , activation='relu' ))

# adding the second hidden layer
classifier.add(Dense( units = 30 , kernel_initializer= 'uniform' , activation='relu' ))

#adding the output layer
classifier.add(Dense( units = 1 , kernel_initializer= 'uniform' , activation='sigmoid' ))   # if the output has more than two categories than use the 'softmax, instead of sigmoid


#compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'] ,)

#fitting the ANN to the training set
classifier.fit(dataset , y , batch_size=10 , epochs=5)


#********************************************training complete*******************************************


test = pd.read_csv('test.csv')
test = test.merge(resource , how= 'left' , on = 'id')
passenger = pd.DataFrame()
passenger['id'] = test['id']


test = test.iloc[: , 2:]
test = test.drop('project_essay_1' ,1)
test = test.drop('project_essay_2',1)
test = test.drop('project_essay_3',1)
test = test.drop('project_essay_4',1)
test = test.drop('project_title',1)
test = test.drop('project_resource_summary',1)
year = list()
month = list()
k = 0
for k in range(0,78035):
   year.append(test.iloc[k ,2].split('-')[0])
   month.append(test.iloc[k ,2].split('-')[1])
test['year'] = year   
test['month'] = month   
test = test.drop('project_submitted_datetime',1)
test['cost'] = test['quantity']*test['price']
test = test.drop('quantity',1)
test = test.drop('price',1)


#dealing with misssing data
test = test.fillna(test['teacher_prefix'].value_counts().index[0])
test = test.fillna(test['school_state'].value_counts().index[0])
test = test.fillna(test['project_grade_category'].value_counts().index[0])
test = test.fillna(test['project_subject_categories'].value_counts().index[0])
test = test.fillna(test['project_subject_subcategories'].value_counts().index[0])
test = test.fillna(test['teacher_number_of_previously_posted_projects'].value_counts().index[0])
test = test.fillna(test['year'].value_counts().index[0])
test = test.fillna(test['month'].value_counts().index[0])
test.iloc[:,2].fillna(test.iloc[: , 8].mean() , inplace = True)

from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
le_2 = LabelEncoder()
le_3 = LabelEncoder()
le_4 = LabelEncoder()
le_5 = LabelEncoder()
le_6 = LabelEncoder()
le_7 = LabelEncoder()
test.iloc[:,0] = le_1.fit_transform(test.iloc[:,0])
test.iloc[:,1] = le_2.fit_transform(test.iloc[:,1])
test.iloc[:,2] = le_3.fit_transform(test.iloc[:,2])
test.iloc[:,3] = le_4.fit_transform(test.iloc[:,3])
test.iloc[:,4] = le_5.fit_transform(test.iloc[:,4])
test.iloc[:,6] = le_6.fit_transform(test.iloc[:,6])
test.iloc[:,7] = le_7.fit_transform(test.iloc[:,7])


test = onehotencoder_1.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_2.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_3.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_4.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_5.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_6.transform(test).toarray()
test = test[: , 1:]
test = onehotencoder_7.transform(test).toarray()
test = test[: , 1:]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test = sc.fit_transform(test)


y_pred = classifier.predict(test)

passenger['project_is_approved'] = y_pred

passenger.to_csv('final.csv' , index = False)