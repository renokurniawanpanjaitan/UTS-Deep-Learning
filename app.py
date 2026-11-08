from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from inference_utils import (decide, find_target_layer, load_assets,
                             make_gradcam, overlay, to_tensor)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model.keras"
CLASSES_PATH = BASE / "models" / "class_names.json"
CONFIG_PATH = BASE / "models" / "inference_config.json"

st.set_page_config(page_title="Deteksi Kematangan Pisang", page_icon="🍌", layout="centered")


@st.cache_resource(show_spinner="Memuat model...")
def get_assets():
    return load_assets(str(MODEL_PATH), str(CLASSES_PATH), str(CONFIG_PATH))


model, class_names, img_size, threshold = get_assets()

st.title("🍌 Deteksi Kematangan Pisang")
st.write(
    "Unggah foto pisang untuk mengetahui tingkat kematangannya. "
    "Aplikasi menampilkan probabilitas tiap kelas, peta Grad-CAM (bagian gambar yang "
    "paling memengaruhi keputusan), dan pesan **TIDAK YAKIN** bila keyakinan model "
    f"di bawah ambang ({threshold:.2f})."
)

file = st.file_uploader("Pilih foto (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "webp"])

if file is None:
    st.info("Belum ada foto. Unggah foto pisang untuk memulai.")
    st.stop()

image = Image.open(file).convert("RGB")
x = to_tensor(np.asarray(image), img_size)
probs = model.predict(x[None, ...], verbose=0)[0]
label, conf, sure = decide(probs, class_names, threshold)

col1, col2 = st.columns(2)
col1.image(image, caption="Foto yang diunggah", use_container_width=True)

if find_target_layer(model) is not None:
    cam = make_gradcam(model, x, class_index=int(np.argmax(probs)))
    col2.image(overlay(np.asarray(x), cam), caption="Grad-CAM (merah/kuning = area penting)",
               use_container_width=True)
else:
    col2.warning("Model ini tidak punya lapisan konvolusi, Grad-CAM tidak tersedia.")

st.subheader("Hasil prediksi")
if sure:
    st.success(f"Prediksi: **{label}** (keyakinan {conf:.1%})")
else:
    st.warning(
        f"**TIDAK YAKIN.** Dugaan terkuat: {label} ({conf:.1%}), di bawah ambang "
        f"{threshold:.0%}. Coba foto yang lebih jelas atau cek manual."
    )

st.subheader("Probabilitas per kelas")
st.bar_chart({c: float(p) for c, p in zip(class_names, probs)}, horizontal=True)
