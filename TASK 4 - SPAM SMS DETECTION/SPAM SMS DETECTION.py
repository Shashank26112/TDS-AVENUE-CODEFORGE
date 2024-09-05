import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the correct encoding
df = pd.read_csv(r"C:/Users/hp/Desktop/ML internship/TASK 4 - SPAM SMS DETECTION/spam.csv", encoding='ISO-8859-1')

# Show basic info and the last few rows
df.info()
df.tail()

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns for better readability
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df.rename(columns={'v1': 'Classification', 'v2': 'SMS Text'}, inplace=True)

# Display the first few rows
df.head()

# Visualize the distribution of the classifications
sns.countplot(data=df, x='Classification')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.title('Distribution of SMSs')
plt.show()

# Encode the labels
df['Classification'] = encoder.fit_transform(df['Classification'])
df.head()

# Check and remove duplicates
df.duplicated().sum()
df = df.drop_duplicates(keep='first')

# Calculate the number of characters in each SMS
df['num_char'] = df['SMS Text'].apply(len)
df.head()

# Descriptive statistics for spam (1) and ham (0) messages
subset_spam = df[df['Classification'] == 1]  # Spam messages
subset_ham = df[df['Classification'] == 0]   # Ham messages

print(subset_spam['Classification'].describe())  # Description for spam
print(subset_ham['Classification'].describe())   # Description for ham

# General dataset description
df.describe()

# Calculate mean number of characters for each class
mean_num_char = df.groupby('Classification')['num_char'].mean()
print(mean_num_char)

# Calculate the number of words per SMS
df['num_words'] = df['SMS Text'].apply(lambda x: len(nltk.word_tokenize(x)))
df.head()

# Calculate mean number of words for each class
mean_num_words = df.groupby('Classification')['num_words'].mean()
print(mean_num_words)

# Calculate the number of sentences per SMS
df['num_sentences'] = df['SMS Text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df.head()

# Calculate mean number of sentences for each class
mean_num_sentences = df.groupby('Classification')['num_sentences'].mean()
print(mean_num_sentences)

# Visualize distribution of the number of characters in spam and ham messages
plt.figure(figsize=(14, 8))
sns.histplot(df[df['Classification'] == 0]['num_char'], color='blue', label='Ham', kde=True)
sns.histplot(df[df['Classification'] == 1]['num_char'], color='red', label='Spam', kde=True)
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.legend()
plt.show()