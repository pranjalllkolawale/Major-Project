import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "deepfake_detector_final.keras"  # use your fine-tuned model
ORIGINAL_VIDEOS_DIR = r"C:\Users\HP\Desktop\MajorProject\dataset\original"
FAKE_VIDEOS_DIR = r"C:\Users\HP\Desktop\MajorProject\dataset\fake"
TEST_VIDEO = r"C:\Users\HP\Desktop\MajorProject\videos\01__kitchen_still.mp4"
IMG_SIZE = (224, 224)

# ----------------------------
# FRAME EXTRACTION
# ----------------------------
def extract_frames(video_path, max_frames=80, img_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

def load_frames_from_folder(folder_path, max_videos=2, max_frames=40, img_size=(224,224)):
    X = []
    for i, file in enumerate(os.listdir(folder_path)):
        if not (file.endswith(".mp4") or file.endswith(".avi")):
            continue
        if i >= max_videos:
            break
        path = os.path.join(folder_path, file)
        frames = extract_frames(path, max_frames, img_size)
        X.extend(frames)
    return np.array(X)

# ----------------------------
# CALIBRATE THRESHOLD
# ----------------------------
def calibrate_threshold(model, real_dir, fake_dir, max_frames=80):
    def mean_score(folder):
        frames = load_frames_from_folder(folder, max_videos=2, max_frames=max_frames)
        preds = model.predict(preprocess_input(frames), verbose=0)
        return np.mean(preds) if len(preds) > 0 else 0

    real_mean = mean_score(real_dir)
    fake_mean = mean_score(fake_dir)
    threshold = (real_mean + fake_mean) / 2

    print("\n⚙️ Threshold Calibration:")
    print(f"   Avg REAL Score = {real_mean:.4f}")
    print(f"   Avg FAKE Score = {fake_mean:.4f}")
    print(f"   ✅ Recommended Threshold = {threshold:.4f}\n")
    return threshold

# ----------------------------
# PREDICT VIDEO
# ----------------------------
def predict_video(video_path, model, threshold=0.5, frames_to_sample=80, img_size=(224,224)):
    frames = extract_frames(video_path, max_frames=frames_to_sample, img_size=img_size)
    if frames.size == 0:
        print(f"⚠️ No frames found in video: {video_path}")
        return

    frames = preprocess_input(frames)
    preds = model.predict(frames, verbose=0)
    mean_score = np.mean(preds)
    result = "REAL" if mean_score > threshold else "FAKE"

    print(f"🎥 Video: {os.path.basename(video_path)}")
    print(f"🧾 Result: {result}")
    print(f"📈 Mean Prediction Score: {mean_score:.4f}")
    print(f"📊 Frame Predictions (first 10): {(preds > threshold).astype(int).flatten()[:10]} ...")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print(f"✅ Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)

    threshold = calibrate_threshold(model, ORIGINAL_VIDEOS_DIR, FAKE_VIDEOS_DIR)

    predict_video(TEST_VIDEO, model, threshold=threshold)