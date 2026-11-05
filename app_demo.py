import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_model.keras")
CLASSES_PATH = os.path.join(BASE, "models", "class_names.json")
CONFIG_PATH = os.path.join(BASE, "models", "inference_config.json")

missing = [p for p in (MODEL_PATH, CLASSES_PATH) if not os.path.exists(p)]
if missing:
    print("\n=== FILE MODEL BELUM ADA ===")
    for p in missing:
        print(" - tidak ditemukan:", p)
    print("\nTaruh best_model.keras dan class_names.json (hasil notebook) di folder 'models'.")
    print("File inference_config.json (hasil notebook Fitur_Unik) bersifat opsional.")
    sys.exit(1)

import gradio as gr
import numpy as np

from inference_utils import decide, load_assets, make_gradcam, overlay, to_tensor

model, NAMES, SIZE, THRESHOLD = load_assets(MODEL_PATH, CLASSES_PATH, CONFIG_PATH)


def analyze(image):
    if image is None:
        return {}, None, "Unggah foto pisang terlebih dahulu."
    x = to_tensor(image, SIZE)
    probs = model.predict(x[None, ...], verbose=0)[0]
    label, conf, sure = decide(probs, NAMES, THRESHOLD)

    try:                                   # Fitur unik 1: Grad-CAM
        cam = make_gradcam(model, x, class_index=int(np.argmax(probs)))
        cam_img = overlay(x, cam)
    except Exception:
        cam_img = None

    if sure:                               # Fitur unik 2: confidence threshold
        msg = f"Prediksi: {label.upper()} (keyakinan {conf * 100:.1f}%)"
    else:
        msg = (f"TIDAK YAKIN (keyakinan tertinggi {conf * 100:.1f}% < ambang {THRESHOLD * 100:.0f}%). "
               "Foto mungkin bukan pisang atau kurang jelas. Mohon foto ulang atau cek manual.")
    return {n: float(p) for n, p in zip(NAMES, probs)}, cam_img, msg


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(label="Unggah foto pisang"),
    outputs=[
        gr.Label(num_top_classes=len(NAMES), label="Probabilitas tiap kelas"),
        gr.Image(label="Grad-CAM (area yang menjadi dasar keputusan model)"),
        gr.Textbox(label="Hasil keputusan"),
    ],
    title="Deteksi Kematangan Pisang (CNN)",
    description="Proyek UTS Deep Learning. Kelas: " + ", ".join(NAMES) +
                ". Dilengkapi Grad-CAM dan penolakan prediksi bila model tidak yakin.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
