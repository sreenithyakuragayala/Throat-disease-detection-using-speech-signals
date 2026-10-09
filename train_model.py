import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
import random
import json

# -------------------------------
# Parameters
# -------------------------------
DATASET_DIR = "dataset"
CLASSES = ["healthy_voices", "diseased_voices"] # Consistent class order
SAMPLE_RATE = 16000
MAX_LEN = 160   # frames (time steps)
N_MELS = 64     # mel-spectrogram channels

# -------------------------------
# Feature Extraction with Augmentation
# -------------------------------
def extract_features(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y)
    y = y / (np.max(np.abs(y)) + 1e-9)  # normalize waveform

    # Augmentation
    if augment:
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 3))
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.005, len(y))
            y = y + noise

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=256, n_fft=512)
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # shape: (frames, N_MELS)

    # Pad or truncate
    if mel_db.shape[0] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, MAX_LEN - mel_db.shape[0]), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:MAX_LEN, :]

    return mel_db

# -------------------------------
# Load Dataset and Balance Classes
# -------------------------------
X, y = [], []
file_paths = []
class_counts = {}

# Gather all file paths and their labels
for label, category in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, category)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
    file_paths.extend([(f, label) for f in files])
    class_counts[category] = len(files)

print("Class counts before balancing:", class_counts)

# Identify the minority class and its target count
min_class = min(class_counts, key=class_counts.get)
max_count = max(class_counts.values())

# Oversample the minority class with augmentation
oversampled_paths = []
for path, label in file_paths:
    oversampled_paths.append((path, label))
    if CLASSES[label] == min_class:
        while len(oversampled_paths) < max_count:
            oversampled_paths.append((path, label))

random.shuffle(oversampled_paths)
print(f"✅ Dataset balanced with oversampling. Total samples: {len(oversampled_paths)}")

# Extract features from the balanced dataset
for path, label in oversampled_paths:
    features = extract_features(path, augment=True)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"✅ Final dataset shape: {X.shape[0]} samples, {X.shape[1:]} feature shape")

# -------------------------------
# Train-test split + Standardization
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features (zero mean, unit variance)
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / (std + 1e-9)
X_test = (X_test - mean) / (std + 1e-9)

# Save normalization parameters for Flask inference
os.makedirs("ml_models", exist_ok=True)
np.save("ml_models/feature_mean.npy", mean)
np.save("ml_models/feature_std.npy", std)
print("✅ Saved mean & std for Flask inference")

# -------------------------------
# Compute Class Weights
# -------------------------------
weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# -------------------------------
# Build CNN + BiLSTM Model
# -------------------------------
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(MAX_LEN, N_MELS)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5), # Increased dropout
    Bidirectional(LSTM(64)),
    Dropout(0.4), # Increased dropout

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# -------------------------------
# Train Model
# -------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("ml_models/best_cnn_bilstm.h5", save_best_only=True, monitor="val_accuracy")
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=8,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# Evaluate and Save Metrics
# -------------------------------
y_pred = model.predict(X_test).argmax(axis=1)

# Generate and save the classification report
report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "accuracy": report["accuracy"],
    "healthy_voices": {
        "precision": report["healthy_voices"]["precision"],
        "recall": report["healthy_voices"]["recall"],
        "f1-score": report["healthy_voices"]["f1-score"]
    },
    "diseased_voices": {
        "precision": report["diseased_voices"]["precision"],
        "recall": report["diseased_voices"]["recall"],
        "f1-score": report["diseased_voices"]["f1-score"]
    },
    "confusion_matrix": cm.tolist()
}

metrics_path = "ml_models/model_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
    
print("✅ Model metrics saved to ml_models/model_metrics.json")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASSES))
print("\nConfusion Matrix:\n")
print(cm)

# -------------------------------
# Save Final Model
# -------------------------------
model.save("ml_models/final_cnn_bilstm.h5")
print("✅ Model saved to ml_models/final_cnn_bilstm.h5")