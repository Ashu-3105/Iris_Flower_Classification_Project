# Importing required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#  Importing dataset
dataset = load_iris()

# Extracting Features and target columns
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['Target'])

# Splitting into Training and testing sets
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3, random_state=42)

# Bulding and training the model

def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train,y_train)
    return trained_model




