"""Fungsi bersama untuk inferensi: preprocessing, Grad-CAM, dan keputusan berbasis confidence threshold.

Dipakai oleh: app_demo.py, predict.py, dan notebook Fitur_Unik.
Preprocessing di sini IDENTIK dengan pipeline training (resize -> bagi 255).
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_assets(model_path, classes_path, config_path=None, default_threshold=0.70):
    """Muat model, nama kelas, ukuran gambar, dan ambang keyakinan (threshold)."""
    model = keras.models.load_model(model_path)
    with open(classes_path) as f:
        meta = json.load(f)
    threshold = default_threshold
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            threshold = float(json.load(f).get("threshold", default_threshold))
    return model, meta["class_names"], meta["img_size"], threshold


def to_tensor(image, size):
    """Terima path file ATAU array/PIL, kembalikan tensor float32 (size, size, 3) skala 0-1."""
    if isinstance(image, str):
        img = tf.io.decode_image(tf.io.read_file(image), channels=3, expand_animations=False)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:                      # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3]                     # buang kanal alpha bila ada
        img = tf.convert_to_tensor(arr)
    img = tf.image.resize(img, (size, size))
    return tf.cast(img, tf.float32) / 255.0


def decide(probs, class_names, threshold):
    """Fitur unik 2: tolak prediksi jika keyakinan tertinggi < threshold."""
    k = int(np.argmax(probs))
    conf = float(probs[k])
    return class_names[k], conf, conf >= threshold


def find_target_layer(model):
    """Lapisan aktivasi ReLU terakhir (keluaran blok konvolusi terakhir) untuk Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.ReLU):
            return layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam(model, x, class_index=None, layer_name=None):
    """Fitur unik 1: Grad-CAM. x = tensor (H, W, 3) skala 0-1. Mengembalikan heatmap (H, W) skala 0-1."""
    target = layer_name or find_target_layer(model)
    if target is None:
        raise ValueError("Model tidak punya lapisan konvolusi; Grad-CAM hanya untuk CNN.")
    with tf.GradientTape() as tape:
        h = tf.convert_to_tensor(x)[None, ...]
        conv_out = None
        for layer in model.layers:
            h = layer(h, training=False)
            if layer.name == target:
                conv_out = h
        if class_index is None:
            class_index = int(tf.argmax(h[0]))
        score = h[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                       # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (x.shape[0], x.shape[1]))[..., 0]
    return cam.numpy()


def overlay(x, heatmap, alpha=0.4):
    """Tumpuk heatmap berwarna di atas gambar. Mengembalikan array uint8 (H, W, 3)."""
    import matplotlib
    heat = matplotlib.colormaps["jet"](heatmap)[..., :3]
    out = (1 - alpha) * np.asarray(x) + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype("uint8")
