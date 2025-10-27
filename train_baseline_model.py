import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset
try:
    df = pd.read_csv('ear_dataset.csv')
except FileNotFoundError:
    print("Error: ear_dataset.csv not found. Please run create_dataset.py first.")
    exit()

# Drop any rows with missing values, just in case
df.dropna(inplace=True)

# 2. Explore the Data (Visualization)
print("Creating data distribution plot...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='label', y='ear', data=df)
plt.title('EAR Distribution for Drowsy (1) vs. Non-Drowsy (0)')
plt.ylabel('Eye Aspect Ratio (EAR)')
plt.xlabel('Label')
plt.xticks([0, 1], ['Non-Drowsy', 'Drowsy'])
plt.show()

# 3. Prepare Data for Training
# X contains our feature (the EAR value)
# y contains our target (the label: 0 or 1)
X = df[['ear']]  # Using double brackets to keep it as a DataFrame
y = df['label']

# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split complete.")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Train the Logistic Regression Model
print("\nTraining a Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate the Model
print("\nEvaluating the model on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)