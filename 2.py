import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------
# DATASET PATHS
# ---------------------------
train_dir = r"frames\train"
val_dir = r"frames\val"

# ---------------------------
# PARAMETERS
# ---------------------------
img_size = (224, 224)
batch_size = 32
epochs = 20

# ---------------------------
# DATA GENERATORS
# ---------------------------
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=img_size,
batch_size=batch_size,
class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
val_dir,
target_size=img_size,
batch_size=batch_size,
class_mode="binary"
)

# ---------------------------
# BUILD MODEL
# ---------------------------
print("🔗 Loading EfficientNetB0 with ImageNet weights...")
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze base model layers first
for layer in base_model.layers[:-40]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# ---------------------------
# CALLBACKS
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

# ---------------------------
# TRAIN PHASE 1
# ---------------------------
print("\n🚀 Phase 1: Training top layers...")
history1 = model.fit(
train_generator,
validation_data=val_generator,
epochs=10,
callbacks=[early_stop, lr_reduce]
)

# ---------------------------
# TRAIN PHASE 2 (Fine-tuning)
# ---------------------------
print("\n🔧 Phase 2: Fine-tuning full model...")
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
loss="binary_crossentropy", metrics=["accuracy"])

history2 = model.fit(
train_generator,
validation_data=val_generator,
epochs=10,
callbacks=[early_stop, lr_reduce]
)

# ---------------------------
# SAVE MODEL
# ---------------------------
model.save("deepfake_detector_finetuned.h5")
print("💾 Model saved as deepfake_detector_finetuned.h5")

# ---------------------------
# EVALUATE MODEL
# ---------------------------
val_loss, val_acc = model.evaluate(val_generator)
print(f"✅ Final Validation Accuracy: {val_acc:.2f}")