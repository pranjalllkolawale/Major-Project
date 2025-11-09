import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ----------------------------
# PATHS
# ----------------------------
MODEL_PATH = "deepfake_detector_finetuned_quick.keras"
FINAL_MODEL_PATH = "deepfake_detector_final.keras"
ORIGINAL_VIDEOS_DIR = r"C:\Users\HP\Desktop\MajorProject\dataset\original"
FAKE_VIDEOS_DIR = r"C:\Users\HP\Desktop\MajorProject\dataset\fake"
TEST_VIDEO = r"C:\Users\HP\Desktop\MajorProject\videos\01__kitchen_still.mp4"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5

# ----------------------------
# UTILS
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

def load_frames_from_folder(folder_path, label=None, max_videos=20, max_frames=20, img_size=(224, 224)):
    X, y = [], []
    for i, file in enumerate(os.listdir(folder_path)):
        if not (file.endswith(".mp4") or file.endswith(".avi")):
            continue
        if i >= max_videos:
            break
        path = os.path.join(folder_path, file)
        frames = extract_frames(path, max_frames, img_size)
        X.extend(frames)
        if label is not None:
            y.extend([label] * len(frames))
    return np.array(X), np.array(y)

# ----------------------------
# LOAD MODEL
# ----------------------------
print(f"✅ Loading model from: {MODEL_PATH}")
base_model = load_model(MODEL_PATH, compile=False)

# Add custom head if needed
if not hasattr(base_model, "layers") or not isinstance(base_model.layers[-1], Dense):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
else:
    model = base_model

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# FINE-TUNING
# ----------------------------
print("📂 Loading frames for fine-tuning...")
real_X, real_y = load_frames_from_folder(ORIGINAL_VIDEOS_DIR, label=1, max_videos=10)
fake_X, fake_y = load_frames_from_folder(FAKE_VIDEOS_DIR, label=0, max_videos=10)

X_train = np.concatenate([real_X, fake_X], axis=0)
y_train = np.concatenate([real_y, fake_y], axis=0)

print(f"✅ Fine-tuning dataset shape: {X_train.shape}, Labels: {y_train.shape}")

X_train = preprocess_input(X_train)

print("🔧 Fine-tuning model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model.save(FINAL_MODEL_PATH)
print(f"💾 Fine-tuned model saved as: {FINAL_MODEL_PATH}")

# ----------------------------
# THRESHOLD CALIBRATION
# ----------------------------
def calibrate_threshold(model, real_dir, fake_dir, max_frames=100):
    def get_mean_score(folder):
        frames = load_frames_from_folder(folder, max_videos=2, max_frames=max_frames)[0]
        preds = model.predict(preprocess_input(frames), verbose=0)
        return np.mean(preds)
    
    real_score = get_mean_score(real_dir)
    fake_score = get_mean_score(fake_dir)
    threshold = (real_score + fake_score) / 2
    print(f"\n⚙️ Threshold calibration:")
    print(f"  Avg REAL score = {real_score:.4f}")
    print(f"  Avg FAKE score = {fake_score:.4f}")
    print(f"  ✅ Recommended threshold = {threshold:.4f}\n")
    return threshold

THRESHOLD = calibrate_threshold(model, ORIGINAL_VIDEOS_DIR, FAKE_VIDEOS_DIR)

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_video(video_path, model, frames_to_sample=80, img_size=(224,224), threshold=0.5):
    frames = extract_frames(video_path, max_frames=frames_to_sample, img_size=img_size)
    if frames.size == 0:
        print(f"⚠️ No frames to predict for video: {video_path}")
        return

    frames = preprocess_input(frames)
    preds = model.predict(frames, verbose=0)
    mean_score = np.mean(preds)
    result = "REAL" if mean_score > threshold else "FAKE"

    print(f"\n🎥 Video: {os.path.basename(video_path)}")
    print(f"🧾 Result: {result}")
    print(f"📈 Mean prediction score: {mean_score:.4f}")
    print(f"📊 Frame predictions (first 10): {(preds > threshold).astype(int).flatten()[:10]} ...")

# ----------------------------
# TEST
# ----------------------------
predict_video(TEST_VIDEO, model, threshold=THRESHOLD)
