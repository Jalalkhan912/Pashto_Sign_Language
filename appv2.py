import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import joblib
import pygame
from PIL import Image, ImageDraw, ImageFont

# --------- Load Traced Model and Scaler ----------
model = torch.jit.load("gesture_classifier_traced.pt")
model.eval()

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------- Pashto Labels Mapping ----------
pashto_labels = {
    0: 'ا', 1: 'آ', 2: 'ع', 3: 'ب', 4: 'چ', 5: 'د', 6: 'ډ', 7: 'ځ', 8: 'ف',
    9: 'ګ', 10: 'ږ', 11: 'ھ', 12: 'غ', 13: 'ه', 14: 'ء', 15: 'ح', 16: 'ج',
    17: 'ک', 18: 'ښ', 19: 'خ', 20: 'ل', 21: 'م', 22: 'ڼ', 23: 'ن', 24: 'پ', 
    25: 'ق', 26: 'ر', 27: 'ړ', 28: 'ص', 29: 'ې', 30: 'س', 31: 'ث', 32: 'ش', 
    33: 'ت', 34: 'ط', 35: 'څ', 36: 'ټ', 37: 'و', 38: 'ی', 39: 'ض', 40: 'ذ', 
    41: 'ز', 42: 'ژ', 43: 'ظ'
}

# --------- Audio Mapping ----------
audio_map = {
    0: "SignLanguageAudio/Alef.wav",
    1: "SignLanguageAudio/Alef_mad.wav",
    2: "SignLanguageAudio/Ayn.wav",
    3: "SignLanguageAudio/Bey.wav",
    4: "SignLanguageAudio/Che.wav",
    5: "SignLanguageAudio/Daal.wav",
    6: "SignLanguageAudio/Ddaal.wav",
    7: "SignLanguageAudio/Dzey.wav",
    8: "SignLanguageAudio/Fe.wav",
    9: "SignLanguageAudio/Gaaf.wav",
    10: "SignLanguageAudio/Ge.wav",
    11: "SignLanguageAudio/Ger_de_he.wav",
    12: "SignLanguageAudio/Ghayn.wav",
    13: "SignLanguageAudio/Halwa_He.wav",
    14: "SignLanguageAudio/Hamza.wav",
    15: "SignLanguageAudio/He.wav",
    16: "SignLanguageAudio/Jeem.wav",
    17: "SignLanguageAudio/Kaaf.wav",
    18: "SignLanguageAudio/Kheen.wav",
    19: "SignLanguageAudio/Khey.wav",
    20: "SignLanguageAudio/Laam.wav",
    21: "SignLanguageAudio/Meem.wav",
    22: "SignLanguageAudio/Nnoon.wav",
    23: "SignLanguageAudio/Noon.wav",
    24: "SignLanguageAudio/Pey.wav",
    25: "SignLanguageAudio/Qaaf.wav",
    26: "SignLanguageAudio/Rey.wav",
    27: "SignLanguageAudio/Rrey.wav",
    28: "SignLanguageAudio/Saad.wav",
    29: "SignLanguageAudio/Sakhta_ye.wav",
    30: "SignLanguageAudio/Seen.wav",
    31: "SignLanguageAudio/Sey.wav",
    32: "SignLanguageAudio/Sheen.wav",
    33: "SignLanguageAudio/Tey.wav",
    34: "SignLanguageAudio/Toey.wav",
    35: "SignLanguageAudio/Tsey.wav",
    36: "SignLanguageAudio/Ttey.wav",
    37: "SignLanguageAudio/Wow.wav",
    38: "SignLanguageAudio/Ye.wav",
    39: "SignLanguageAudio/Zaad.wav",
    40: "SignLanguageAudio/Zaal.wav",
    41: "SignLanguageAudio/Ze.wav",
    42: "SignLanguageAudio/Zhe.wav",
    43: "SignLanguageAudio/Zoey.wav",
}

# --------- Font Configuration ----------
font_path = "Pashto Standard Fonts v02.0.0/Bahij Aban-Bold.ttf"  # Ensure this font file is available

# --------- Initialize Pygame for Audio ----------
pygame.mixer.init()
current_audio = None  # Track the currently playing audio

# --------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# --------- Normalize Keypoints Relative to Bounding Box ----------
def normalize_landmarks(landmarks):
    keypoints = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    x_coords, y_coords = keypoints[:, 0], keypoints[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    width = x_max - x_min if x_max - x_min != 0 else 1e-6
    height = y_max - y_min if y_max - y_min != 0 else 1e-6

    keypoints[:, 0] = (x_coords - x_min) / width
    keypoints[:, 1] = (y_coords - y_min) / height

    return keypoints.flatten()

# --------- Draw Pashto Text Function ----------
def draw_pil_text(frame, text, position, font_path, font_size=48, color=(0, 255, 0)):
    """ Draw Pashto text on an OpenCV frame using PIL. """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert OpenCV to PIL
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font file not found. Using default PIL font.")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

# --------- Audio Management Function ----------
def play_audio_for_class(class_id):
    global current_audio
    if class_id in audio_map:
        if current_audio != class_id:
            try:
                pygame.mixer.stop()  # Stop any playing audio
                pygame.mixer.music.load(audio_map[class_id])
                pygame.mixer.music.play(-1)  # Loop indefinitely
                current_audio = class_id
                print(f"Playing audio for class {class_id}: {pashto_labels.get(class_id, 'Unknown')}")
            except pygame.error as e:
                print(f"Error playing audio: {e}")

def stop_audio():
    global current_audio
    pygame.mixer.music.stop()
    current_audio = None

# --------- Main Camera Loop ----------
cap = cv2.VideoCapture(0)
last_detection_time = time.time()
no_detection_threshold = 1.0  # Stop audio after 1 second of no detection

print("Starting Pashto Sign Language Recognition...")
print("Press 'q' to quit")

while True:
    start_time = time.time()

    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    detection_made = False

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
            class_id = pred.item()
            confidence = conf.item()

            # Convert class_id to match pashto_labels if needed
            # (Assuming your model outputs match the pashto_labels indices)
            
            # Only proceed if confidence is above threshold
            if confidence > 0.7:  # Adjust threshold as needed
                detection_made = True
                last_detection_time = time.time()
                
                # Get Pashto label
                pashto_label = pashto_labels.get(class_id, "Unknown")
                
                # Get English label for display
                try:
                    english_label = label_encoder.inverse_transform([class_id])[0]
                except:
                    english_label = f"Class_{class_id}"

                # 4. Display predictions with Pashto text
                cv2.putText(frame, f"English: {english_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw Pashto text using PIL
                if pashto_label != "Unknown":
                    frame = draw_pil_text(frame, f"Pashto: {pashto_label}", (10, 100), font_path, font_size=48, color=(255, 0, 0))
                
                # Play corresponding audio
                play_audio_for_class(class_id)

    # Stop audio if no detection for a while
    if not detection_made and (time.time() - last_detection_time) > no_detection_threshold:
        stop_audio()

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)  # Add small value to avoid division by zero
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Add instructions
    cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Pashto Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.quit()
print("Application closed successfully!")