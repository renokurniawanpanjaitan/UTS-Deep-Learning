# Proyek 1 UTS Deep Learning: Klasifikasi Kematangan Pisang dengan CNN

Folder ini mengikuti format pengumpulan `UTS DEEP LEARNING`.
**Ganti nama folder** menjadi nama kelompok dan kelas Anda sebelum dikumpulkan.

## Isi folder
| Berkas | Fungsi |
|---|---|
| `KlasifikasiKematanganPisang.ipynb` | Notebook utama: data, EDA, tf.data, MLP, CNN, optimasi, evaluasi, inferensi (dokumentasi + source code) |
| `predict.py` | Prediksi dari gambar baru lewat terminal |
| `app_demo.py` | Demo web (Gradio) untuk video presentasi |
| `requirements.txt` | Daftar library |
| `dataset/` | Tempat dataset (isi sendiri, lihat petunjuk di dalamnya) |
| `models/` | Terisi otomatis setelah notebook dijalankan |

## Langkah menjalankan
1. **Instal library**: `pip install -r requirements.txt` (atau pakai Google Colab dengan GPU: Runtime > Change runtime type > GPU).
2. **Jalankan notebook** dari sel pertama sampai terakhir. Hasil grafik tersimpan di `outputs/`, model di `models/`, log di `logs/`.
3. **Demo**: `python app_demo.py` atau `python predict.py foto.jpg`.

## Pemetaan ke CPMK
| CPMK | Bagian notebook |
|---|---|
| 1. Konsep deep learning, tensor, TensorFlow | Bagian 4 (pipeline `tf.data`, demo operasi tensor) |
| 2. Perceptron, MLP, aktivasi, backpropagation, gradient descent | Bagian 4 (demo Dense manual & GradientTape) dan Bagian 5 (baseline MLP) |
| 3. Optimasi & regularisasi | Bagian 6 dan 7 (loss, Adam vs RMSProp, Dropout, BatchNorm, ReduceLROnPlateau, EarlyStopping, monitoring, augmentation) |
| 4. CNN dasar | Bagian 5 (3 blok Conv2D + Pooling) dan Bagian 8 (evaluasi) |

