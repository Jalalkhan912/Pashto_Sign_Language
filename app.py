import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib

# --------- Load Traced Model and Scaler ----------
model = torch.jit.load("gesture_classifier_traced.pt")
model.eval()

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# --------- Normalize Keypoints Relative to Bounding Box ----------
def normalize_landmarks(landmarks):
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    width = x_max - x_min if x_max - x_min != 0 else 1e-6
    height = y_max - y_min if y_max - y_min != 0 else 1e-6

    keypoints[:, 0] = (x_coords - x_min) / width
    keypoints[:, 1] = (y_coords - y_min) / height

    return keypoints.flatten()


# --------- Camera Loop ----------
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 1:
        hand_landmarks = result.multi_hand_landmarks[0]

        # 1. Draw landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 2. Extract and normalize
        norm_keypoints = normalize_landmarks(hand_landmarks)
        input_scaled = scaler.transform([norm_keypoints]).astype(np.float32)
        input_tensor = torch.from_numpy(input_scaled)

        # 3. Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            label = label_encoder.inverse_transform(pred.numpy())[0]
            confidence = conf.item()

        # 4. Display prediction
        cv2.putText(frame, f"Class: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
