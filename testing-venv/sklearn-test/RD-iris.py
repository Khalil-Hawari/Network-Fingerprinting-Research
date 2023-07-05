### Multiclass Classification using Random Forest on Scikit-Learn Library

#Importing Libraries
import numpy as np
np.__version__
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
import joblib
import os

cwd = os.getcwd()
iris_file =  os.sep.join([cwd, 'testing-venv', 'sklearn-test', 'iris', 'iris.data.csv'])
print('cwd:', iris_file)



#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv(iris_file, header = None)
# dataset = pd.read_csv('testing-venv\sklearn-test\iris\iris.data.csv', header = None)
#Renaming the columns
dataset.columns = ['sepal length in cm', 'sepal width in cm','petal length in cm','petal width in cm','species']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())



#Creating the dependent variable class
# All this really does is convert species names to numbers, since SK cannot work with strings
factor = pd.factorize(dataset['species'])
dataset.species = factor[0]
definitions = factor[1]
print(dataset.species.head())
print(definitions)