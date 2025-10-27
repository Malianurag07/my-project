import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate

# --- 1. Load the Sequence Dataset ---
try:
    df = pd.read_csv('ear_sequences.csv')
except FileNotFoundError:
    print("Error: ear_sequences.csv not found. Please run create_sequences.py first.")
    exit()

# --- 2. Create the New Personalized Feature ---
print("Creating personalized baseline feature...")

# The original sequences of 30 EAR values
sequences = df.drop('label', axis=1).values
labels = df['label'].values

# Calculate the new feature for each sequence
baseline_features = []
for seq in sequences:
    # Baseline is the average of the first 10 frames
    baseline_ear = np.mean(seq[:10])
    # Current state is the average of the last 5 frames
    current_ear = np.mean(seq[-5:])
    # The feature is the deviation from the baseline
    deviation = current_ear - baseline_ear
    baseline_features.append(deviation)

baseline_features = np.array(baseline_features).reshape(-1, 1)

# --- 3. Prepare Data for the Multi-Input Model ---
# Reshape sequence data for the LSTM
sequences_reshaped = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

# Split all data (sequences, baseline feature, and labels)
# We need to split them all together to keep them aligned
X_seq_train, X_seq_test, X_base_train, X_base_test, y_train, y_test = train_test_split(
    sequences_reshaped, baseline_features, labels, test_size=0.2, random_state=42
)

print("Data prepared with two types of input.")

# --- 4. Build the Final (Multi-Input) Model ---
print("\nBuilding the final model...")

# Input layer for the sequence data (for the LSTM)
sequence_input = Input(shape=(sequences_reshaped.shape[1], 1), name='sequence_input')
lstm_out = LSTM(64)(sequence_input)

# Input layer for our single baseline feature
baseline_input = Input(shape=(1,), name='baseline_input')

# Concatenate (merge) the output of the LSTM with the baseline feature
merged = concatenate([lstm_out, baseline_input])

# Add a dense layer to interpret the merged features
x = Dense(32, activation='relu')(merged)
# Final output layer
output = Dense(1, activation='sigmoid')(x)

# Create the model with two inputs and one output
model = Model(inputs=[sequence_input, baseline_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train the Model ---
print("\nTraining the final model...")
history = model.fit(
    {'sequence_input': X_seq_train, 'baseline_input': X_base_train},
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=({'sequence_input': X_seq_test, 'baseline_input': X_base_test}, y_test)
)

# --- 6. Evaluate the Final Model ---
print("\nEvaluating the final model on the test set...")
loss, accuracy = model.evaluate(
    {'sequence_input': X_seq_test, 'baseline_input': X_base_test},
    y_test
)

print("-" * 30)
print(f"Final Personalized Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)
# --- Add this line at the end of train_final_model.py ---
model.save('drowsiness_model.keras')
print("\nModel saved to drowsiness_model.keras")
from tensorflow.keras.models import load_model

# Load the model created with the newer Keras version
model = load_model('drowsiness_model.keras')

# Save the model in the older, widely compatible .h5 format
model.save('drowsiness_model.h5')

print("Model successfully converted to drowsiness_model.h5")