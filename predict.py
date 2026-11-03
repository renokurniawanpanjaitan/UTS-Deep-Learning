"""Inferensi kematangan pisang dari satu atau beberapa gambar.

Contoh:
    python predict.py foto1.jpg foto2.jpg
    python predict.py foto.jpg --model models/best_model.keras --classes models/class_names.json
"""
import argparse
import json

import tensorflow as tf
from tensorflow import keras


def load_image(path, img_size):
    # Preprocessing identik dengan pipeline training (decode -> resize -> normalisasi 0-1)
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description="Prediksi tingkat kematangan pisang")
    ap.add_argument("images", nargs="+", help="path gambar")
    ap.add_argument("--model", default="models/best_model.keras")
    ap.add_argument("--classes", default="models/class_names.json")
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    with open(args.classes) as f:
        meta = json.load(f)
    names, size = meta["class_names"], meta["img_size"]

    for path in args.images:
        probs = model.predict(load_image(path, size)[None, ...], verbose=0)[0]
        k = int(probs.argmax())
        detail = ", ".join(f"{n}={p * 100:.1f}%" for n, p in zip(names, probs))
        print(f"{path}: {names[k].upper()} ({probs[k] * 100:.1f}%)  [{detail}]")


if __name__ == "__main__":
    main()
