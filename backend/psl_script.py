from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

sequence_length = 60
# Mediapipe models
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = np.array(sequence)
        padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
        sequence = np.vstack((sequence, padding))
    return sequence

def preprocess_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    while cap.isOpened() and len(keypoints_sequence) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        keypoints_sequence.append(keypoints)
    cap.release()
    keypoints_sequence = pad_sequence(keypoints_sequence, sequence_length)
    return np.expand_dims(np.array(keypoints_sequence), axis=0)

def main(video_path: str):
    # Parameters
    actions = ['ATM', 'Atmosphere', 'Beach', 'Bedroom', 'Car', 'Food Order', 'Food1', 'Hospital', 'Hotel', 'Kitchen', 'Mosque', 'Park']
    model_path = 'PSL_Model_v8(81).h5'
    # print("Entered main .......................................")
    # Load model
    model = load_model(model_path)

    # Mediapipe Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        keypoints_sequence = preprocess_video(video_path, holistic)

    # Predict
    if keypoints_sequence.shape[1:] == (sequence_length, 1662):  # Ensure correct shape
        prediction = model.predict(keypoints_sequence)
        predicted_action = actions[np.argmax(prediction)]
        # return predicted_action
        print(f"Predicted action: {predicted_action}")
    else:
        return "Error: Input shape mismatch with the model!"
