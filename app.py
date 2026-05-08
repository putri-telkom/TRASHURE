import os
import base64
import cv2
import numpy as np
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "dataset"
MODEL_PATH = "waste_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ["organik", "anorganik", "B3"]


def ensure_dataset_structure():
    os.makedirs(DATA_DIR, exist_ok=True)
    for label in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)


def load_trained_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model tidak ditemukan. Letakkan waste_classifier.h5 di folder proyek atau latih model terlebih dahulu."
        )
    return load_model(MODEL_PATH)


def preprocess_image_bytes(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Tidak dapat membaca gambar dari kamera.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)


def predict_image(image_bytes, model):
    image = preprocess_image_bytes(image_bytes)
    predictions = model.predict(image)
    index = int(np.argmax(predictions[0]))
    label = CLASSES[index]
    confidence = float(predictions[0][index])
    return label, confidence


def save_sample(image_bytes, label):
    if label not in CLASSES:
        raise ValueError("Label tidak valid.")

    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Tidak dapat membaca gambar dari kamera.")

    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    destination = os.path.join(DATA_DIR, label, filename)
    cv2.imwrite(destination, frame)
    return destination


def build_model(num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def prepare_data_generators():
    ensure_dataset_structure()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    valid_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, valid_generator


def train_model(epochs=10):
    train_gen, valid_gen = prepare_data_generators()

    if train_gen.samples == 0 or valid_gen.samples == 0:
        raise ValueError(
            "Data tidak cukup. Pastikan setiap kelas memiliki beberapa gambar di folder dataset/organik, dataset/anorganik, dataset/B3."
        )

    model = build_model(len(CLASSES))
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        verbose=0,
    )
    model.save(MODEL_PATH)

    return {
        "train_samples": train_gen.samples,
        "valid_samples": valid_gen.samples,
        "accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
    }


def dataset_stats():
    ensure_dataset_structure()
    counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in CLASSES}
    return counts


app = Flask(__name__)
model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        try:
            model = load_trained_model()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Permintaan tidak valid. Kirim data gambar base64."}), 400

    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        label, confidence = predict_image(image_bytes, model)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/collect", methods=["POST"])
def collect():
    data = request.get_json(silent=True)
    if not data or "image" not in data or "label" not in data:
        return jsonify({"error": "Permintaan tidak valid. Kirim image dan label."}), 400

    label = data["label"]
    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        save_sample(image_bytes, label)
        counts = dataset_stats()
        return jsonify({"success": True, "counts": counts})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 10))

    try:
        result = train_model(epochs=epochs)
        global model
        model = None
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/stats")
def stats():
    return jsonify(dataset_stats())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
