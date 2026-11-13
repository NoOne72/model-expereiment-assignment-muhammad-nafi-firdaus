# AI Model Experiment & Evaluation

## 1. Project Overview

Proyek ini membandingkan dua pendekatan untuk melakukan **sentiment analysis** terhadap ulasan pelanggan e-commerce berbahasa Indonesia:

1. **Classic Machine Learning:** TF-IDF (unigram dan bigram) + Logistic Regression menggunakan Scikit-learn.
2. **Large Language Model:** Gemini AI API dengan pendekatan zero-shot prompting.

Tujuan eksperimen adalah mengukur performa kedua pendekatan dan menilai trade-off dari sisi implementasi, kecepatan inference, potensi biaya, serta keterbatasannya sebagai dasar rekomendasi teknis.

## 2. Problem Statement

Tim produk e-commerce membutuhkan fitur yang dapat mengklasifikasikan ulasan pelanggan menjadi sentimen **positif** atau **negatif** secara otomatis. Eksperimen ini menjadi evaluasi awal sebelum pendekatan yang paling sesuai dikembangkan lebih lanjut.

### Objective
Membandingkan performa model klasifikasi teks Scikit-learn yang dilatih menggunakan dataset dengan hasil klasifikasi Gemini AI API yang menggunakan prompting tanpa training.

### Target / Label
- `positif`: ulasan yang menunjukkan kepuasan, pujian, atau pengalaman baik.
- `negatif`: ulasan yang menunjukkan keluhan, kekecewaan, kritik, atau pengalaman buruk.

### Batasan dan asumsi
- Dataset yang digunakan adalah dataset assignment `customer_reviews_sentiment.csv`; file mentah tidak dimodifikasi.
- Kolom utama yang digunakan adalah `review_text` dan `sentiment`.
- Label dinormalisasi menjadi `positif` dan `negatif`, kemudian dipetakan ke angka 1 dan 0.
- Teks duplikat digabungkan untuk menghindari pengulangan teks identik dalam evaluasi.
- Kedua pendekatan dievaluasi pada 40 teks unik yang sama.
- Model klasik menggunakan stratified 5-fold out-of-fold prediction.
- Gemini menggunakan zero-shot prompt dengan temperature 0.0 dan structured JSON output.
- Training Deep Learning dari nol tidak termasuk ruang lingkup assignment.

## 3. Dataset dan Preprocessing

Dataset awal berisi **200 baris ulasan**. Setelah validasi, normalisasi, dan penggabungan teks duplikat, diperoleh **40 teks unik**:
- 20 ulasan negatif
- 20 ulasan positif

Preprocessing yang diterapkan:
1. Memeriksa nilai kosong pada teks dan label.
2. Menghapus baris dengan teks atau label kosong.
3. Merapikan spasi dan menormalkan label menjadi huruf kecil.
4. Melakukan cleaning teks berupa lowercase, penghapusan tanda baca/simbol, dan perapian spasi.
5. Mempertahankan stopword dan kata negasi karena kata seperti `tidak`, `belum`, dan `sangat` dapat memengaruhi polaritas sentimen.
6. Menggabungkan teks ulasan identik dan menyimpan informasi jumlah kemunculannya.
7. Membuat pembagian stratified 5-fold untuk evaluasi model klasik.

Dataset mentah ditempatkan di `data/customer_reviews_sentiment.csv` dan tidak diubah langsung oleh notebook.

## 4. Experiment Setup

### 4.1 Scikit-learn

- Feature extraction: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)`.
- Classifier: `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.
- Evaluation: stratified 5-fold out-of-fold prediction.
- TF-IDF berada di dalam pipeline sehingga vocabulary dan IDF di-fit hanya menggunakan data training pada setiap fold.

### 4.2 Gemini AI API

- Model: `gemini-3.5-flash` (sesuai konfigurasi notebook saat eksperimen).
- Prompting: zero-shot.
- Temperature: `0.0`, dipilih untuk mengurangi variasi output.
- Output: JSON terstruktur dengan ID ulasan dan label `positif` atau `negatif`.
- Batch: 40 teks unik dalam satu pemanggilan berhasil.
- Retry: maksimal 5 retry tambahan apabila terjadi error API atau respons tidak lengkap.
- API key dibaca dari environment variable `GEMINI_API_KEY`, bukan ditulis langsung di notebook.

## 5. Evaluation Results

Kedua pendekatan dibandingkan pada **40 teks yang sama**. Tidak ada output Gemini yang hilang atau tidak valid pada run yang tercatat.

### 5.1 Overall metrics

| Metric | Scikit-learn | Gemini zero-shot |
|---|---:|---:|
| Accuracy | 0.775 | 1.000 |
| Precision (macro) | 0.781 | 1.000 |
| Recall (macro) | 0.775 | 1.000 |
| F1-Score (macro) | 0.774 | 1.000 |

Gemini memperoleh nilai lebih tinggi pada seluruh metrik di eksperimen ini. Hasil ini berlaku untuk kumpulan 40 teks unik yang diuji dan tidak otomatis menggambarkan performa pada data produksi.

### 5.2 Confusion matrix

Baris menunjukkan label aktual dan kolom menunjukkan hasil prediksi.

**Scikit-learn**

| Aktual \ Prediksi | Negatif | Positif |
|---|---:|---:|
| Negatif | 14 | 6 |
| Positif | 3 | 17 |

Model klasik mengklasifikasikan 31 dari 40 teks dengan benar dan melakukan 9 kesalahan.

**Gemini zero-shot**

| Aktual \ Prediksi | Negatif | Positif |
|---|---:|---:|
| Negatif | 20 | 0 |
| Positif | 0 | 20 |

Gemini mengklasifikasikan seluruh 40 teks dengan benar pada satu run yang tercatat.

### 5.3 Per-fold Scikit-learn

| Fold | Jumlah data uji | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 0 | 8 | 0.750 | 0.750 |
| 1 | 8 | 0.750 | 0.750 |
| 2 | 8 | 1.000 | 1.000 |
| 3 | 8 | 0.750 | 0.750 |
| 4 | 8 | 0.625 | 0.564 |
| **Mean** | **40** | **0.775** | **0.763** |

Standar deviasi Macro-F1 antar-fold adalah 0.155. Perbedaan skor antar-fold menunjukkan performa model klasik belum stabil pada setiap subset data.

**Catatan metodologi:** Scikit-learn menghasilkan prediksi out-of-fold melalui pelatihan pada empat fold dan pengujian pada fold tersisa. Gemini memprediksi seluruh 40 teks dalam satu batch, lalu hasilnya dikelompokkan menurut penanda fold untuk pelaporan. Pengelompokan tersebut bukan proses cross-validation Gemini. Perbandingan dilakukan pada teks yang sama, tetapi kedua prosedur evaluasi tidak identik secara mekanisme.

## 6. Operational Results

| Aspek | Scikit-learn | Gemini zero-shot |
|---|---|---|
| Training | Perlu training pada setiap fold | Tidak memerlukan training |
| Inference | Lokal | API eksternal |
| Waktu training | Rata-rata sekitar 0.007–0.012 detik per fold | Tidak ada |
| Waktu prediksi | Sekitar 0.31–0.52 ms per teks pada fold | 228.05 detik untuk satu batch 40 teks |
| Token | Tidak menggunakan token API | 768 prompt tokens, 394 output tokens, 2,051 total tokens menurut metadata |
| Biaya | Tidak ada biaya API per prediksi | Berpotensi berbayar sesuai penggunaan token |
| Ketersediaan | Dapat dijalankan lokal setelah dependensi tersedia | Memerlukan internet dan ketersediaan layanan API |

Latency Gemini adalah waktu keseluruhan batch, bukan pengukuran latency individual setiap teks. Biaya API belum dihitung karena harga input dan output token belum diisi pada konfigurasi eksperimen.

## 7. Trade-off dan Limitations

### Scikit-learn
**Kelebihan**
- Inference lokal sangat cepat setelah model tersedia.
- Tidak membutuhkan biaya API per request.
- Pipeline dan proses training dapat dikontrol serta direproduksi.
- Cocok menjadi baseline ringan untuk klasifikasi berulang.

**Keterbatasan**
- Performa bergantung pada jumlah dan variasi data berlabel.
- TF-IDF lebih berfokus pada frekuensi kata dan n-gram, sehingga konteks kalimat kompleks, ironi, dan negasi dapat sulit ditangkap.
- Terdapat 9 kesalahan prediksi dari 40 teks.
- Macro-F1 antar-fold memiliki variasi yang cukup besar.

### Gemini zero-shot
**Kelebihan**
- Tidak memerlukan training khusus.
- Prompt dan format output dapat disiapkan dengan relatif sederhana.
- Pada eksperimen ini seluruh 40 teks berhasil diklasifikasikan dengan benar.

**Keterbatasan**
- Skor sempurna diperoleh hanya pada 40 teks unik dan satu run.
- Konsistensi antar-run belum diuji (`N_RUNS = 1`).
- Batch membutuhkan sekitar 228.05 detik.
- Bergantung pada koneksi internet, ketersediaan API, batas penggunaan, dan biaya token.
- Hasil pada dataset kecil belum menjamin generalisasi ke ulasan baru.

### Keterbatasan dataset dan evaluasi
Dataset awal berjumlah 200 baris, tetapi deduplikasi menyisakan 40 teks unik. Jumlah contoh independen yang kecil membatasi variasi bahasa dan konteks yang dapat dievaluasi. Karena itu, hasil ini merupakan eksperimen awal, bukan validasi final untuk produksi.

## 8. Technical Recommendation

Berdasarkan hasil yang tersedia, **Gemini zero-shot direkomendasikan sebagai kandidat utama untuk validasi lanjutan**, karena memperoleh Accuracy, Precision, Recall, dan Macro-F1 sebesar 1.000 pada 40 teks yang diuji. Model klasik tetap dipertahankan sebagai baseline lokal karena inference-nya jauh lebih cepat dan tidak memerlukan biaya API per prediksi.

Gemini belum direkomendasikan untuk langsung digunakan di produksi. Sebelum keputusan final, perlu dilakukan:
1. Pengujian pada dataset baru yang lebih besar dan lebih beragam.
2. Pengulangan inference Gemini beberapa kali untuk mengukur konsistensi.
3. Pengukuran latency dengan metode yang konsisten dan pada beban yang representatif.
4. Pengisian harga token sesuai tarif model yang digunakan untuk menghitung biaya aktual.
5. Penetapan ambang batas performa, latency, dan biaya berdasarkan kebutuhan produk.

Dengan demikian, keputusan saat ini adalah melanjutkan validasi Gemini sebagai kandidat dengan performa awal terbaik, sambil mempertahankan Scikit-learn sebagai alternatif lokal yang efisien. Pemilihan solusi produksi harus menunggu hasil pengujian lanjutan dan analisis biaya yang lebih lengkap.

## 9. Project Structure

```text
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   └── model_comparison_summary.png   # opsional
├── README.md
└── requirements.txt
```

## 10. How to Run

### Prerequisites
- Python 3.10+ (gunakan versi yang kompatibel dengan library di `requirements.txt`).
- VS Code dengan ekstensi Jupyter, atau Jupyter Notebook.
- Gemini API key untuk menjalankan eksperimen LLM.

### Setup environment

```bash
# Clone repository
git clone <URL_REPOSITORY>
cd assignment-model-experiment-nama-peserta

# Buat virtual environment
python -m venv .venv

# Aktivasi pada Windows PowerShell
.venv\Scripts\Activate.ps1

# Aktivasi pada macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Gemini API key

**Windows PowerShell**
```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

**macOS/Linux**
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Jangan menaruh API key langsung di notebook atau mengunggahnya ke GitHub.

### Run notebook

1. Pastikan dataset tersedia di `data/customer_reviews_sentiment.csv`.
2. Buka `notebook/experiment_notebook.ipynb` melalui VS Code/Jupyter.
3. Pilih kernel dari virtual environment yang sudah dibuat.
4. Jalankan cell secara berurutan dari Section 2 sampai Section 8.
5. Pastikan `GEMINI_API_KEY` sudah tersedia sebelum menjalankan bagian Gemini.
6. Periksa hasil evaluasi, confusion matrix, trade-off, dan rekomendasi yang ditampilkan.

Notebook menggunakan relative path `../data/customer_reviews_sentiment.csv`, sehingga notebook perlu berada di dalam folder `notebook/` seperti struktur di atas.
