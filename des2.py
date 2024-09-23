import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'Enjoy sports.csv'
dataset = pd.read_csv(dataset_path)

# Drop unnecessary 'Day' column
dataset = dataset.drop(columns=['Day'])

# Apply Label Encoding to categorical features
label_encoder = LabelEncoder()
for column in dataset.columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Separate the features (X) and target variable (y)
X = dataset.drop(columns=['Decision'])  # Features
y = dataset['Decision']  # Target

# Create and train the Decision Tree model
clf = DecisionTreeClassifier(criterion='entropy')  # Using entropy for ID3 algorithm
clf.fit(X, y)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=list(X.columns), class_names=['No', 'Yes'], filled=True)  # Convert feature_names to list
plt.show()
