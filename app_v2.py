import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import joblib
from PIL import ImageFont, ImageDraw, Image
import pygame
import os
pygame.mixer.init()

# --------- Load Traced Model and Scaler ----------
model = torch.jit.load("gesture_classifier_traced.pt")
model.eval()

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------- Pashto Mapping and Audio Map ----------
label_to_pashto = {
    0: 'ا', 1: 'آ', 2: 'ع', 3: 'ب', 4: 'چ', 5: 'د', 6: 'ډ', 7: 'ځ', 8: 'ف',
    9: 'ګ', 10: 'ږ', 11: 'ھ', 12: 'غ', 13: 'ه', 14: 'ء', 15: 'ح', 16: 'ج',
    17: 'ک', 18: 'ښ', 19: 'خ', 20: 'ل', 21: 'م', 22: 'ڼ', 23: 'ن', 24: 'پ', 
    25: 'ق', 26: 'ر', 27: 'ړ', 28: 'ص', 29: 'ې', 30: 'س', 31: 'ث', 32: 'ش', 
    33: 'ت', 34: 'ط', 35: 'څ', 36: 'ټ', 37: 'و', 38: 'ی', 39: 'ض', 40: 'ذ', 
    41: 'ز', 42: 'ژ', 43: 'ظ'
}

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

# --------- Load Pashto Font ----------
font_path = "Pashto Standard Fonts v02.0.0/Bahij Aban-Bold.ttf"
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Pashto font not found at {font_path}")
pashto_font = ImageFont.truetype(font_path, 48)

# --------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# --------- Normalize Keypoints Relative to Bounding Box ----------
def normalize_landmarks(landmarks):
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    x_coords, y_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    width = x_max - x_min if x_max - x_min != 0 else 1e-6
    height = y_max - y_min if y_max - y_min != 0 else 1e-6
    keypoints[:, 0] = (x_coords - x_min) / width
    keypoints[:, 1] = (y_coords - y_min) / height
    return keypoints.flatten()

# --------- Audio Playback with Threading ----------
last_played = None
last_played = None
def play_audio(label):
    global last_played
    path = audio_map.get(label)
    if path and os.path.exists(path):
        if last_played != label:
            last_played = label
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()


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
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Process
        norm_keypoints = normalize_landmarks(hand_landmarks)
        input_scaled = scaler.transform([norm_keypoints]).astype(np.float32)
        input_tensor = torch.from_numpy(input_scaled)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            label = label_encoder.inverse_transform(pred.numpy())[0]
            confidence = conf.item()

        # Pashto Label
        pashto_text = label_to_pashto.get(label, label)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), pashto_text, font=pashto_font, fill=(0, 255, 0))
        draw.text((10, 60), f"Confidence: {confidence:.2f}", font=pashto_font, fill=(255, 255, 0))
        frame = np.array(img_pil)

        # Play Sound
        play_audio(label)

    # Display
    cv2.imshow("Pashto Gesture App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
hands.close()
