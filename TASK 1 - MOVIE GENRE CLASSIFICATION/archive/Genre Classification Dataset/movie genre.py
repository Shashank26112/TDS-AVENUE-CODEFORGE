import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_path = "C:/Users/hp/Desktop/ML internship/TASK 1 - MOVIE GENERE CLASSIFICATION/archive/Genre Classification Dataset/train_data.txt"
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genere', 'Description'], engine='python')
train_data.columns
train_data.head(10)
train_data.isnull().sum()
train_data.info()
train_data.describe()
test_path = "C:/Users/hp/Desktop/ML  internship/TASK 1 - MOVIE GENERE CLASSIFICATION/archive/Genre Classification Dataset/test_data.txt"
test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
test_data.columns
test_data.head(20)
test_data.isnull().sum()
test_data.info()
test_data.describe()

# Load test data solution
test_solution_path = "C:/Users/hp/Desktop/ML internship/TASK 1 - MOVIE GENERE CLASSIFICATION/archive/Genre Classification Dataset/test_data_solution.txt"
test_solution = pd.read_csv(test_solution_path, sep=":::", names=['Id', 'Title', 'Genere', 'Description'], engine="python")
test_solution.head(10)
genere_counts = train_data['Genre'].value_counts()
genere_counts = genre_counts.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=genere_counts.index, y=genere_counts.values, palette='viridis')
plt.title('Distribution of Generes in Training Data', fontsize=16)
plt.xlabel('Genere', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
