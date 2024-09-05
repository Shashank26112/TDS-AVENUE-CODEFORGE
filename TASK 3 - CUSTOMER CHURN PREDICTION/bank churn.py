from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/hp/Desktop/ML internship/TASK 3 - CUSTOMER CHURN PREDICTION/Churn_Modelling.csv")
data.info()
data["Geography"].unique()
data.describe()
data.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
data.head()
labelencoder=LabelEncoder()
data['Gender']=labelencoder.fit_transform(data['Gender'])
data['Geography']=labelencoder.fit_transform(data['Geography'])
data.head()
data.dtypes
x = data.drop(columns='Exited')  # droping coulmn from the feature
y = data['Exited']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=40)


def churn_bar_chart(feature):
    Exited = data[data['Exited'] == 1][feature].value_counts()
    not_Exited = data[data['Exited'] == 0][feature].value_counts()

    df = pd.DataFrame([Exited, not_Exited])
    df.index = ['Exited', ' Not Exited']

    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title(f"Churn by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()