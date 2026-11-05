import argparse
import os

import numpy as np

from inference_utils import decide, load_assets, make_gradcam, overlay, to_tensor

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Prediksi tingkat kematangan pisang")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=os.path.join(BASE, "models", "best_model.keras"))
    ap.add_argument("--classes", default=os.path.join(BASE, "models", "class_names.json"))
    ap.add_argument("--config", default=os.path.join(BASE, "models", "inference_config.json"))
    ap.add_argument("--gradcam", action="store_true")
    a = ap.parse_args()

    model, names, size, thr = load_assets(a.model, a.classes, a.config)
    for path in a.images:
        x = to_tensor(path, size)
        probs = model.predict(x[None, ...], verbose=0)[0]
        label, conf, sure = decide(probs, names, thr)
        detail = ", ".join(f"{n}={p * 100:.1f}%" for n, p in zip(names, probs))
        status = label.upper() if sure else f"TIDAK YAKIN (terbaik: {label}, ambang {thr:.2f})"
        print(f"{path}: {status} ({conf * 100:.1f}%)  [{detail}]")
        if a.gradcam:
            from PIL import Image
            out = os.path.splitext(path)[0] + "_gradcam.png"
            cam = make_gradcam(model, x, class_index=int(np.argmax(probs)))
            Image.fromarray(overlay(x, cam)).save(out)
            print("  Grad-CAM disimpan:", out)


if __name__ == "__main__":
    main()
