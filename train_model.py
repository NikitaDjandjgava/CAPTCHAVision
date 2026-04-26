import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models

LETTER_DIR = "extracted_letter_images"

X, y = [], []

for label in sorted(os.listdir(LETTER_DIR)):
    folder = os.path.join(LETTER_DIR, label)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (20, 20))
        X.append(img)
        y.append(label)

X = np.array(X, dtype="float32") / 255.0
X = X[..., np.newaxis]  # shape: (N, 20, 20, 1)

lb = LabelBinarizer()
y_enc = lb.fit_transform(y)
num_classes = len(lb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=0)

model = models.Sequential([
    layers.Input(shape=(20, 20, 1)),
    layers.Conv2D(20, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(50, (5, 5), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(500, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=32)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"test accuracy: {acc:.4f}")

model.save("captcha_model.keras")
with open("model_labels.pkl", "wb") as f:
    pickle.dump(lb, f)

print("saved model and labels")