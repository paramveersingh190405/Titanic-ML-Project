import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv('tested.csv')

# Step 2: Initial inspection
print("Initial Data Shape:", df.shape)
print(df.info())
print(df.head())

# Step 3: Handle missing values
df = df.dropna(axis=0, how='any')  # Drop rows with any missing values (or use fillna if needed)

# Step 4: Remove duplicates
df = df.drop_duplicates()

# Step 5: Correct data types
# Convert numerical columns stored as strings
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].astype(float)
        except:
            pass  # Keep as object if conversion fails

# Step 6: Feature Engineering (example: extract title from name if it exists)
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Example: Create age category
if 'Age' in df.columns:
    bins = [0, 12, 18, 35, 60, np.inf]
    labels = ['Child', 'Teen', 'Adult', 'Middle Age', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Step 7: Encode categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Step 8: Define features and target
# Try to guess target column (you can manually change it too)
possible_targets = ['Survived', 'target', 'Target', 'Outcome']
target_col = [col for col in df.columns if col in possible_targets]

if target_col:
    y = df[target_col[0]]
    X = df.drop(target_col[0], axis=1)
else:
    raise Exception("❌ Target column not found. Please rename it to 'target', 'Survived', etc.")

# Step 9: Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 10: Feature Scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 11: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Step 12: Evaluation
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
