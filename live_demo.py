import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import math

# --- Helper Functions (from our previous scripts) ---
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# --- Load the Trained Model ---
print("Loading the trained model...")
# NEW LINE:
model = tf.keras.models.load_model('drowsiness_model.h5')
print("Model loaded successfully.")

# --- Initialize Webcam and MediaPipe ---
cap = cv2.VideoCapture(0) # 0 is the default webcam
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# --- Real-time Processing Variables ---
SEQUENCE_LENGTH = 30
ear_sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_text = "Connecting..."
prediction_color = (0, 255, 0) # Green for Alert

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # --- Feature Extraction ---
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR
        left_eye_pts = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE_INDICES]
        
        v1_left = distance(left_eye_pts[1], left_eye_pts[5])
        v2_left = distance(left_eye_pts[2], left_eye_pts[4])
        h_left = distance(left_eye_pts[0], left_eye_pts[3])
        ear_left = (v1_left + v2_left) / (2.0 * h_left)
        
        v1_right = distance(right_eye_pts[1], right_eye_pts[5])
        v2_right = distance(right_eye_pts[2], right_eye_pts[4])
        h_right = distance(right_eye_pts[0], right_eye_pts[3])
        ear_right = (v1_right + v2_right) / (2.0 * h_right)
        
        current_ear = (ear_left + ear_right) / 2.0
        
        # Add the current EAR to our sequence
        ear_sequence.append(current_ear)

        # --- Prediction ---
        # We need a full sequence of 30 frames to make a prediction
        if len(ear_sequence) == SEQUENCE_LENGTH:
            # Convert deque to numpy array
            seq_array = np.array(ear_sequence)
            
            # Create the personalized baseline feature
            baseline_ear = np.mean(seq_array[:10])
            current_ear_avg = np.mean(seq_array[-5:])
            deviation = current_ear_avg - baseline_ear
            
            # Reshape for model input
            # The model expects a batch, so the shape is (1, timesteps, features)
            input_seq = seq_array.reshape(1, SEQUENCE_LENGTH, 1)
            input_base = np.array([[deviation]]) # Shape (1, 1)
            
            # Make a prediction
            prediction_prob = model.predict([input_seq, input_base])[0][0]
            
            # Update prediction text and color based on the probability
            if prediction_prob > 0.5:
                prediction_text = f"DROWSY ({prediction_prob:.2f})"
                prediction_color = (0, 0, 255) # Red for Drowsy
            else:
                prediction_text = f"ALERT ({prediction_prob:.2f})"
                prediction_color = (0, 255, 0) # Green for Alert

    else:
        # If no face is detected, reset the sequence
        ear_sequence.clear()
        prediction_text = "NO FACE DETECTED"
        prediction_color = (0, 165, 255) # Orange

    # --- Display Output on Frame ---
    cv2.putText(frame, prediction_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, prediction_color, 2)

    cv2.imshow('Driver Drowsiness Detection', frame)

    # --- Exit Condition ---
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()