# 🤟 Pashto Sign Language Recognition App

This is a real-time Pashto sign language recognition application built with Python. It integrates **MediaPipe**, **PyTorch**, **OpenCV**, **Pygame**, and **Pashto phonetics** to recognize hand gestures corresponding to Pashto alphabet letters from live camera feed, display the recognized character in Pashto and English, and play the associated Pashto audio.

---

## 🧠 How It Works

1. **Dataset Creation**:  
   Using MediaPipe, real sign images were processed to extract 2D hand keypoints. These were normalized relative to the bounding box of each hand and used to build a dataset.

2. **Model Training**:  
   A classification model was trained on the hand keypoints using PyTorch and saved as a traced model (`gesture_classifier_traced.pt`). The data was scaled and label-encoded using `scikit-learn`.

3. **Real-Time Inference**:  
   During app execution:
   - A live webcam feed is analyzed using MediaPipe to detect hands and extract keypoints.
   - The trained model classifies the gesture.
   - The corresponding **Pashto character**, **English label**, and **confidence** are displayed.
   - A native **Pashto audio file** is played.
   - Pashto characters are rendered using a custom font.

---

## 📦 Features

- 🔠 **44 Pashto characters** supported
- 🎙️ **Native audio pronunciation** playback
- 🎥 **Live webcam input**
- 🧠 **PyTorch-based gesture classification**
- ✍️ **Pashto and English text rendering**
- 🎨 **PIL font integration** for complex script rendering

---

## 🚀 Requirements

Make sure you have the following installed:

```bash
pip install opencv-python mediapipe torch numpy pygame joblib scikit-learn pillow
```

Also ensure:
- `gesture_classifier_traced.pt` (traced PyTorch model)
- `scaler.pkl` and `label_encoder.pkl` (from your training phase)
- `Pashto Standard Fonts v02.0.0/Bahij Aban-Bold.ttf` (for correct Pashto script rendering)
- `SignLanguageAudio/` folder with all the `.wav` files for each character

---

## 🛠️ Running the App

```bash
python appv2.py
```

- Press **`q`** to quit the application
- Audio will play when a valid gesture is detected and stop automatically when the gesture disappears

---

## 🎯 Folder Structure

```
├── app.py
├── gesture_classifier_traced.pt
├── scaler.pkl
├── label_encoder.pkl
├── Pashto Standard Fonts v02.0.0/
│   └── Bahij Aban-Bold.ttf
├── SignLanguageAudio/
│   ├── Alef.wav
│   ├── Bey.wav
│   └── ... (all 44 Pashto audio files)
```

---

## 📈 Future Improvements

- Add support for **gesture sequences** (e.g., forming words)
- Train with more diverse datasets for higher accuracy
- Build a **mobile version** using MediaPipe and PyTorch Lite
- Integrate **voice-to-sign** translation

---

## 📚 Acknowledgments

- **[MediaPipe](https://mediapipe.dev/)** by Google for hand tracking
- **[PyTorch](https://pytorch.org/)** for gesture classification
- **Pashto Fonts** for accurate script rendering
- **OpenCV** and **Pillow** for real-time display and text drawing

---

## 📝 License

This project is for educational purposes. Please contact the author for reuse, modifications, or commercial use.