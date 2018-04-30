"""
    This code runs a classification task using a vanilla decision-tree
    from the Python-library SkLearn.
"""

# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# Load data
data = pd.read_csv('../../datasets/tree_addhealth.csv')

# Remove NaN's from dataset
data = data.dropna()

# Specify features of dataset
X = data[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

# Specify target
y = data.TREG1

# Split in train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# Create decision tree model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Make predictions
pred = clf.predict(X_test)

# Analyse FP, TP, FN and TN rates
print(confusion_matrix(y_test, pred))
