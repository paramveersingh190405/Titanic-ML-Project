import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('tested.csv')

# Clean columns
df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin'], errors='ignore')

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Feature engineering: Extract title from name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df = df.drop(columns=['Name'])

# Group rare titles
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Other')

# Encode categorical columns
cat_cols = ['Sex', 'Embarked', 'Title']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Drop target column (Survived) since this is unsupervised
features = df.drop(columns=['Survived'])

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# ======================
# 🔹 Apply K-Means Clustering
# ======================
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add clusters to original dataframe
df['Cluster'] = clusters

# ======================
# 🔹 Apply PCA for 2D Visualization
# ======================
pca = PCA(n_components=2)
pca_components = pca.fit_transform(features_scaled)

df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# ======================
# 🔹 Plotting Clusters
# ======================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=80)
plt.title('K-Means Clustering with PCA (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()
