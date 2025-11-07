import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------
# 1️⃣ Load your trained model
# ---------------------------
model_path = "deepfake_detector_finetuned.h5"
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

# ---------------------------
# 2️⃣ Parameters
# ---------------------------
img_size = (224, 224)
batch_size = 16
epochs = 2 # quick fine-tune
frames_to_sample = 50 # frames per video for fine-tuning
threshold = 0.2 # threshold for real/fake prediction

# ---------------------------
# 3️⃣ Extract frames from video
# ---------------------------
def extract_frames(video_path, max_frames=50, img_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ Video has 0 frames: {video_path}")
        cap.release()
        return np.array([])

    step = max(total_frames // max_frames, 1)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frames.append(frame)
        count += 1
    cap.release()
    frames = np.array(frames)
    if len(frames) == 0:
        print(f"⚠️ No frames extracted from video: {video_path}")
    return frames

# ---------------------------
# 4️⃣ Prepare fine-tuning dataset (original videos only)
# ---------------------------
original_videos = [
    r"C:\Users\HP\Desktop\MajorProject\01__kitchen_still.mp4",
    # add more paths if needed
]

X_real = []
y_real = []

for vid in original_videos:
    frames = extract_frames(vid, max_frames=frames_to_sample, img_size=img_size)
    if frames.size > 0:
        X_real.append(frames)
        y_real.append(np.ones(frames.shape[0])) # label 1 = original

if len(X_real) == 0:
    print("⚠️ No frames extracted from any original video. Fine-tuning skipped.")
else:
    X_real = np.concatenate(X_real, axis=0)
    y_real = np.concatenate(y_real, axis=0)
    print(f"✅ Prepared {len(X_real)} frames from original videos for fine-tuning.")

# ---------------------------
# 5️⃣ Fine-tune last layers
# ---------------------------
for layer in model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_real, y_real,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True
)

# ---------------------------
# 6️⃣ Save fine-tuned model
# ---------------------------
fine_tuned_model_path = "deepfake_detector_finetuned_quick.keras"
model.save(fine_tuned_model_path)
print(f"💾 Fine-tuned model saved as: {fine_tuned_model_path}")

# ---------------------------
# 7️⃣ Predict video function
# ---------------------------
def predict_video_quick(video_path, model, frames_to_sample=80, img_size=(224,224), threshold=0.2):
    frames = extract_frames(video_path, max_frames=frames_to_sample, img_size=img_size)
    if frames.size == 0:
        print(f"⚠️ No frames to predict for video: {video_path}")
        return

    preds = model.predict(frames)
    frame_preds = (preds > threshold).astype(int).flatten()
    counts = np.bincount(frame_preds)
    predicted_class = np.argmax(counts)
    result = "REAL" if predicted_class == 1 else "FAKE"

    print(f"\n🎥 Video: {os.path.basename(video_path)}")
    print(f"🧾 Result: {result}")
    print(f"📈 Mean prediction score: {np.mean(preds):.4f}")
    print(f"📊 Frame predictions (first 10): {frame_preds[:10]} ...")

# ---------------------------
# 8️⃣ Example usage
# ---------------------------
sample_video = r"C:\Users\HP\Desktop\MajorProject\01__kitchen_still.mp4"
predict_video_quick(sample_video, model, frames_to_sample=80, img_size=img_size, threshold=threshold)

    