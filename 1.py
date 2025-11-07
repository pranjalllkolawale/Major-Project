import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
def extract_frames(video_path, output_dir, frames_per_video=10, size=(128,128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frames_per_video, 1)

    count, frame_id = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, size)
            save_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame{frame_id}.jpg")
            cv2.imwrite(save_path, frame)
            frame_id += 1
        count += 1
    cap.release()
# Create frame directories
os.makedirs("frames/original", exist_ok=True)
os.makedirs("frames/fake", exist_ok=True)

# Extract frames from original videos
original_dir = r"C:\Users\HP\Downloads\archive (15)\DFD_original sequences"
for video in os.listdir(original_dir):
    extract_frames(os.path.join(original_dir, video), "frames/original")

# Extract frames from manipulated videos
fake_dir = r"C:\Users\HP\Downloads\archive (15)\DFD_manipulated_sequences\DFD_manipulated_sequences"
for video in os.listdir(fake_dir):
    extract_frames(os.path.join(fake_dir, video), "frames/fake")