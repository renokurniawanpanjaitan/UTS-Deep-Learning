# Proyek 1 UTS Deep Learning: Klasifikasi Kematangan Pisang dengan CNN

# Deteksi Kematangan Pisang

Aplikasi Streamlit untuk klasifikasi kematangan pisang (unripe / ripe / overripe) dengan CNN,
dilengkapi dua fitur unik: **Grad-CAM** dan **confidence threshold** (pesan "TIDAK YAKIN").

## Struktur
- `app.py` : aplikasi Streamlit
- `inference_utils.py` : fungsi inferensi, Grad-CAM, dan keputusan threshold
- `models/` : `best_model.keras`, `class_names.json`, `inference_config.json`
- `requirements.txt` : dependensi

## Cara menjalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Notebook
`KlasifikasiKematanganPisang.ipynb` (pelatihan dan evaluasi) dan
`Fitur_Unik_GradCAM_Threshold.ipynb` (Grad-CAM dan threshold).

## Aplikasi online
Tautan: deteksi-kematangan-pisang-i5zkundhm7njm9qqtyceip.streamlit.app



## Pemetaan ke CPMK
| CPMK | Bagian notebook |
|---|---|
| 1. Konsep deep learning, tensor, TensorFlow | Bagian 4 (pipeline `tf.data`, demo operasi tensor) |
| 2. Perceptron, MLP, aktivasi, backpropagation, gradient descent | Bagian 4 (demo Dense manual & GradientTape) dan Bagian 5 (baseline MLP) |
| 3. Optimasi & regularisasi | Bagian 6 dan 7 (loss, Adam vs RMSProp, Dropout, BatchNorm, ReduceLROnPlateau, EarlyStopping, monitoring, augmentation) |
| 4. CNN dasar | Bagian 5 (3 blok Conv2D + Pooling) dan Bagian 8 (evaluasi) |

