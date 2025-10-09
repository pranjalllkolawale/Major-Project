import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = "deepfake_detector_final.keras"
ORIGINAL_VIDEOS_DIR = r"C:\Users\HP\Downloads\archive (15)\DFD_original sequences"
FAKE_VIDEOS_DIR = r"C:\Users\HP\Downloads\archive (15)\DFD_manipulated_sequences\DFD_manipulated_sequences"
IMG_SIZE = (224, 224)

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("✅ Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# -----------------------------
# FRAME EXTRACTION
# -----------------------------
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

# -----------------------------
# CALIBRATE THRESHOLD
# -----------------------------
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

# -----------------------------
# PREDICT VIDEO
# -----------------------------
def predict_video(video_path, model, threshold=0.5, frames_to_sample=80, img_size=(224,224)):
    frames = extract_frames(video_path, max_frames=frames_to_sample, img_size=img_size)
    if frames.size == 0:
        return {"error": f"No frames found in video: {video_path}"}

    frames = preprocess_input(frames)
    preds = model.predict(frames, verbose=0)
    mean_score = float(np.mean(preds))
    result = "REAL" if mean_score > threshold else "FAKE"

    return {
        "filename": os.path.basename(video_path),
        "result": result,
        "mean_score": round(mean_score, 4),
        "threshold_used": round(threshold, 4),
    }

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # 🔹 Recalibrate threshold before prediction (just like your original script)
    threshold = calibrate_threshold(model, ORIGINAL_VIDEOS_DIR, FAKE_VIDEOS_DIR)

    result = predict_video(file_path, model, threshold=threshold)
    return jsonify(result)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
