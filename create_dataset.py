import cv2
import mediapipe as mp
import math
import os
import csv

# --- EAR Calculation Functions (from our previous script) ---
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True)

def calculate_ear(image_path):
    """Calculates the average Eye Aspect Ratio (EAR) for a single image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Left Eye EAR
        left_eye_pts = [landmarks[i] for i in LEFT_EYE_INDICES]
        v1_left = distance(left_eye_pts[1], left_eye_pts[5])
        v2_left = distance(left_eye_pts[2], left_eye_pts[4])
        h_left = distance(left_eye_pts[0], left_eye_pts[3])
        ear_left = (v1_left + v2_left) / (2.0 * h_left)
        
        # Right Eye EAR
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE_INDICES]
        v1_right = distance(right_eye_pts[1], right_eye_pts[5])
        v2_right = distance(right_eye_pts[2], right_eye_pts[4])
        h_right = distance(right_eye_pts[0], right_eye_pts[3])
        ear_right = (v1_right + v2_right) / (2.0 * h_right)
        
        # Average EAR
        avg_ear = (ear_left + ear_right) / 2.0
        return avg_ear
        
    return None # No face found

# --- Main Script to Process Dataset ---

# UPDATE THIS PATH to the folder containing 'Drowsy' and 'Non Drowsy'
DATASET_PATH = "dataset" 
folders = ["Non Drowsy", "Drowsy"]
labels = [0, 1] # 0 for Non Drowsy, 1 for Drowsy

# Prepare to write to a CSV file
csv_file = open('ear_dataset.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['ear', 'label']) # Write header

# Loop through each folder (Non Drowsy, then Drowsy)
for folder, label in zip(folders, labels):
    folder_path = os.path.join(DATASET_PATH, folder)
    
    print(f"Processing folder: {folder}...")
    
    for filename in os.listdir(folder_path):
        # Make sure we're processing image files
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            
            # Calculate EAR
            ear_value = calculate_ear(image_path)
            
            # If a face was detected, write the data to our CSV
            if ear_value is not None:
                writer.writerow([ear_value, label])

print("Processing complete!")
csv_file.close()