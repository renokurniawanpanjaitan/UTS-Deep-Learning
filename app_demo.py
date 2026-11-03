"""
Jalankan:  python app_demo.py
Lalu buka alamat lokal yang muncul (biasanya http://127.0.0.1:7860)
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_model.keras")
CLASSES_PATH = os.path.join(BASE, "models", "class_names.json")

# Cek file model lebih dulu agar pesan error mudah dipahami
missing = [p for p in (MODEL_PATH, CLASSES_PATH) if not os.path.exists(p)]
if missing:
    print("\n=== FILE MODEL BELUM ADA ===")
    for p in missing:
        print(" - tidak ditemukan:", p)
    print("\nModel dibuat setelah notebook (.ipynb) dijalankan sampai selesai.")
    print("Setelah itu, taruh dua file ini di folder 'models' di samping app_demo.py:")
    print("   best_model.keras  dan  class_names.json")
    sys.exit(1)

import gradio as gr
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    meta = json.load(f)
NAMES, SIZE = meta["class_names"], meta["img_size"]


def classify(image):
    if image is None:
        return {}
    x = tf.image.resize(tf.convert_to_tensor(image[..., :3], dtype=tf.float32), (SIZE, SIZE))
    x = x / 255.0
    probs = model.predict(x[None, ...], verbose=0)[0]
    return {n: float(p) for n, p in zip(NAMES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(label="Unggah foto pisang"),
    outputs=gr.Label(num_top_classes=len(NAMES), label="Tingkat kematangan"),
    title="Deteksi Kematangan Pisang (CNN)",
    description="Proyek UTS Deep Learning. Klasifikasi: unripe / ripe / overripe.",
)

if __name__ == "__main__":
    demo.launch()