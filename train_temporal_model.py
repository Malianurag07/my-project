import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Load the Sequence Dataset ---
try:
    df = pd.read_csv('ear_sequences.csv')
except FileNotFoundError:
    print("Error: ear_sequences.csv not found. Please run create_sequences.py first.")
    exit()

# --- 2. Prepare Data for the LSTM ---
# X contains the sequences of 30 EAR values
# y contains the single label for each sequence
X = df.drop('label', axis=1).values
y = df['label'].values

# The LSTM layer expects data to be in a 3D shape: [samples, timesteps, features]
# We have:
#   - samples: the number of sequences
#   - timesteps: 30 (the length of each sequence)
#   - features: 1 (just the EAR value at each timestep)
# We need to reshape X from (num_sequences, 30) to (num_sequences, 30, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data prepared and split.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# --- 3. Build the LSTM Model ---
print("\nBuilding the LSTM model...")
model = Sequential([
    # Input shape is (timesteps, features), which is (30, 1) for us
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    # Output layer for binary classification (Drowsy/Non-Drowsy)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 4. Train the Model ---
print("\nTraining the model...")
# An epoch is one full pass through the training data.
# We will train for 10 epochs.
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test) # We use the test set for validation during training
)


# --- 5. Evaluate the Final Model ---
print("\nEvaluating the final model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test)

print("-" * 30)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)