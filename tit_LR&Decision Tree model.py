import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load dataset
df = pd.read_csv('tested.csv')

# Display initial info
print("Original shape:", df.shape)
print(df.head())

# Drop irrelevant or non-informative columns if they exist
df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin'], errors='ignore')

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Feature Engineering: Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df = df.drop(columns=['Name'])

# Map rare titles to "Other"
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Other')

# Encode categorical columns
cat_cols = ['Sex', 'Embarked', 'Title']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Feature Scaling
scaler = MinMaxScaler()
features = df.drop('Survived', axis=1)
scaled_features = scaler.fit_transform(features)
X = pd.DataFrame(scaled_features, columns=features.columns)
y = df['Survived']

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# =======================
# 🔹 Linear Regression
# =======================
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_preds = lr_model.predict(x_test)

# Convert regression outputs to binary for classification accuracy
lr_preds_binary = [1 if val > 0.5 else 0 for val in lr_preds]

# Evaluation for Linear Regression
print("\n--- Linear Regression Results ---")
print(f"MSE: {mean_squared_error(y_test, lr_preds):.4f}")
print(f"R² Score: {r2_score(y_test, lr_preds):.4f}")
print(f"Classification Accuracy: {accuracy_score(y_test, lr_preds_binary):.4f}")

# =======================
# 🔹 Decision Tree Classifier
# =======================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
dt_preds = dt_model.predict(x_test)

# Evaluation for Decision Tree
print("\n--- Decision Tree Results ---")
print(f"MSE: {mean_squared_error(y_test, dt_preds):.4f}")
print(f"R² Score: {r2_score(y_test, dt_preds):.4f}")
print(f"Classification Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
