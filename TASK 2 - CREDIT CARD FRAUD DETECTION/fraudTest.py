import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Check files in the directory
for dirname, _, filenames in os.walk('C:/Users/hp/Desktop/ML internship/TASK 2 - CREDIT CARD FRAUD DETECTION'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load data
train_path = 'C:/Users/hp/Desktop/ML internship/TASK 2 - CREDIT CARD FRAUD DETECTION/fraudTest.csv'
test_path = 'C:/Users/hp/Desktop/ML internship/TASK 2 - CREDIT CARD FRAUD DETECTION/fraudTest.csv'

try:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
except PermissionError:
    print("Permission denied. Check if the file is open in another application or if you have the necessary permissions.")
except FileNotFoundError:
    print("File not found. Check the path: {train_path}")
except Exception as e:
    print("An error occurred: {e}")

# Display data
print(train_data.head())
print(test_data.head())

# Data details
print(train_data.columns)
print(test_data.columns)

# Drop invalid data
train_data = train_data.drop(columns='Unnamed: 0', errors='ignore')
test_data = test_data.drop(columns='Unnamed: 0', errors='ignore')

# Encoding categorical columns
encoder = LabelEncoder()

categorical_columns = ['merchant', 'category', 'street', 'job', 'trans_num', 'first', 'city', 'state', 'last', 'gender', 'trans_date_trans_time', 'dob']

for column in categorical_columns:
    train_data[column] = encoder.fit_transform(train_data[column].astype(str))
    test_data[column] = encoder.transform(test_data[column].astype(str))  # Use transform for test data

# Splitting data
x_train = train_data.drop(columns='is_fraud')
y_train = train_data['is_fraud']
x_test = test_data.drop(columns='is_fraud')
y_test = test_data['is_fraud']

# Build classification models
modelLR = LogisticRegression(max_iter=1000, random_state=42)
modelRF = RandomForestClassifier(random_state=42)
modelDT = DecisionTreeClassifier(random_state=42)

# Fit models
modelLR.fit(x_train, y_train)
modelRF.fit(x_train, y_train)
modelDT.fit(x_train, y_train)

# Make predictions
predictionsLR = modelLR.predict(x_test)
predictionsRF = modelRF.predict(x_test)
predictionsDT = modelDT.predict(x_test)

# Make evaluations
cmLR = confusion_matrix(y_test, predictionsLR)
cmRF = confusion_matrix(y_test, predictionsRF)
cmDT = confusion_matrix(y_test, predictionsDT)

# Get models accuracy
accuracyLR = accuracy_score(y_test, predictionsLR)
accuracyRF = accuracy_score(y_test, predictionsRF)
accuracyDT = accuracy_score(y_test, predictionsDT)

print(f"Logistic Regression Accuracy: {accuracyLR}")
print(f"Random Forest Accuracy: {accuracyRF}")
print(f"Decision Tree Accuracy: {accuracyDT}")

# Print classification reports
print("Logistic Regression Classification Report:\n", classification_report(y_test, predictionsLR))
print("Random Forest Classification Report:\n", classification_report(y_test, predictionsRF))
print("Decision Tree Classification Report:\n", classification_report(y_test, predictionsDT))
