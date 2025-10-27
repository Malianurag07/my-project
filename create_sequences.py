import pandas as pd
import numpy as np
import csv

# --- Configuration ---
SEQUENCE_LENGTH = 30  # How many frames of EAR data to look at for one prediction
INPUT_CSV = 'ear_dataset.csv'
OUTPUT_CSV = 'ear_sequences.csv'

# --- Load the dataset ---
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found. Please run create_dataset.py first.")
    exit()

print(f"Loaded {len(df)} total measurements.")

# --- Create Sequences ---
sequences = []
labels = []

# Iterate through the dataframe to create overlapping sequences
for i in range(len(df) - SEQUENCE_LENGTH + 1):
    # Get a sequence of EAR values
    ear_sequence = df['ear'].iloc[i:i + SEQUENCE_LENGTH].values
    
    # The label for the sequence is the label of the LAST frame in the sequence
    sequence_label = df['label'].iloc[i + SEQUENCE_LENGTH - 1]
    
    # Store them
    sequences.append(ear_sequence)
    labels.append(sequence_label)

# Convert lists to numpy arrays for easier handling
sequences = np.array(sequences)
labels = np.array(labels)

print(f"Created {len(sequences)} sequences of length {SEQUENCE_LENGTH}.")

# --- Save the new dataset ---
# We'll save it in a way that's easy to load. Each row will be:
# ear_1, ear_2, ..., ear_30, label
header = [f'ear_{i+1}' for i in range(SEQUENCE_LENGTH)] + ['label']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    for i in range(len(sequences)):
        row = list(sequences[i]) + [labels[i]]
        writer.writerow(row)

print(f"Successfully saved sequence data to {OUTPUT_CSV}.")