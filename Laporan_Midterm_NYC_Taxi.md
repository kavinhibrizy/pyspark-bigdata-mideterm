# LAPORAN UJIAN TENGAH SEMESTER (MIDTERM)
## Mata Kuliah: Big Data & AI
**Judul Proyek:** New York City Taxi Fare Prediction menggunakan PySpark

**Disusun Oleh:**
- Calvin Nathaniel Melciades (103012340285)
- Kavin Hibrizy Pradipto Eska (103012340295)

---

### BAB I. PENDAHULUAN

#### 1.1 Latar Belakang
Prediksi tarif taksi merupakan salah satu permasalahan klasik dalam dunia *Machine Learning*. Di kota metropolitan seperti New York City, tarif taksi berfluktuasi secara dinamis yang dipengaruhi oleh jarak tempuh, waktu keberangkatan, dan titik lokasi geografis. Mengingat dataset rekaman taksi NYC berukuran sangat besar (skala Big Data), pemrosesan data konvensional seringkali tidak memadai. Oleh karena itu, pendekatan dengan komputasi terdistribusi diperlukan untuk membangun model yang akurat dan dapat diskalakan (*scalable*).

#### 1.2 Tujuan Proyek
Tujuan utama dari proyek Midterm ini adalah:
1. Membangun model *Machine Learning* untuk memprediksi tarif taksi (`fare_amount`).
2. Menerapkan metodologi *Data Preprocessing* dan *Feature Engineering* pada dataset berskala besar.
3. Melakukan evaluasi, perbandingan, dan optimasi terhadap beberapa algoritma Machine Learning menggunakan framework Apache Spark.

#### 1.3 Teknologi dan Perangkat Lunak
- **Framework Utama:** Apache Spark (PySpark) & PySpark MLlib.
- **Bahasa Pemrograman:** Python (via Jupyter Notebook).
- **Library Tambahan:** Pandas (untuk I/O files), Matplotlib & Seaborn (untuk visualisasi).

---

### BAB II. METODOLOGI DAN PRA-PEMROSESAN DATA

#### 2.1 Deskripsi Data Mentah (Input)
Data yang digunakan bersumber dari kompetisi Kaggle yang terdiri atas file `train.csv` dan `test.csv`. Kolom dasar yang tersedia meliputi: `key`, `fare_amount` (target), `pickup_datetime`, koordinat titik jemput dan turun (*latitude/longitude*), serta `passenger_count`.

#### 2.2 Pembersihan Data (Data Cleaning)
Tahapan awal proses *(Process)* pembersihan meliputi:
- Mengeliminasi baris data yang mengandung nilai *Null* (kosong) dan *Infinity*.
- Melakukan pemfilteran data tak masuk akal (*outliers*), seperti menghapus baris dengan nilai tarif negatif dan menyingkirkan baris dengan koordinat yang jauh berada di luar peta kota New York.

#### 2.3 Rekayasa Fitur (Feature Engineering)
Proses transformasi data mentah menjadi fitur yang dapat dipelajari oleh model:
- **Fitur Waktu (Temporal):** Memecah string `pickup_datetime` menjadi kolom Jam, Hari, Bulan, dan Tahun guna menangkap pola seperti tarif jam macet.
- **Fitur Spasial (Jarak):** Menghitung jarak lurus absolut rute perjalanan (dalam format *miles*) berdasarkan titik jemput dan turun menggunakan fungsi Euclidean matematis.
- **Vektorisasi:** Menggunakan `VectorAssembler` bawaan PySpark untuk menyatukan seluruh matriks input numerik menjadi satu kolom vektor yang diberi nama `features`.

#### 2.4 Sampling Data Latih
**Output:** Menghasilkan DataFrame turunan bernama `train_data_sampled`. Karena keterbatasan RAM pada komputer lokal, kami mengabil *sampling* acak sebesar 10% dari data bersih agar terhindar dari resiko *Out-of-Memory (OOM)*.

---

### BAB III. IMPLEMENTASI DAN PEMODELAN

#### 3.1 Analisis Eksplorasi Data (EDA)
Sebelum tahap komputasi Machine Learning, dilakukan analisis statistik inferensial:
- **Proses:** Menghitung matriks korelasi silang (`Correlation.corr`) antar seluruh variabel numerik.
- **Output:** Dihasilkan **Visualisasi Heatmap**. Grafik ini membuktikan dugaan awal bahwa fitur *distance* (jarak tempuh) memegang nilai korelasi positif tertinggi terhadap `fare_amount`.

#### 3.2 Pemilihan Model
Dua arsitektur model regresi *Ensemble* dipilih untuk menguji akurasi data:
1. **Random Forest Regressor:** Model yang membangun puluhan pohon keputusan paralel dan merata-ratakan prediksinya guna mencegah *overfitting* data. (Parameter awal: `maxDepth=8`, `numTrees=30`).
2. **Gradient Boosting Regressor (GBT):** Model konsekutif di mana setiap pohon berusaha memperbaiki nilai *error* dari kalkulasi pohon sebelumnya. (Parameter awal: `maxDepth=5`, `maxIter=50`).

#### 3.3 Pelatihan Model (Training)
- **Input:** Kolom input `features` dan target `label` disuplai ke dalam kedua model tersebut.
- **Output:** Terciptanya objek memori berisi model cerdas (`rf_model` dan `gbt_model`) yang siap diadu.

---

### BAB IV. HASIL DAN PEMBAHASAN

#### 4.1 Evaluasi Kinerja Model
Setelah prediksi dilakukan pada data latih, metrik kesalahan dihitung menggunakan `RegressionEvaluator`. Berikut adalah tabel komparasi aktual hasil pemrograman:

| Algoritma / Model | RMSE | MAE | R² (R-Squared) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 2.2794 | 1.6298 | 0.6965 |
| **Gradient Boosting** | 2.1684 | 1.5223 | 0.7254 |

**Pembahasan:** Program secara otomatis menetapkan **Gradient Boosting** sebagai juara utama (`best_model`). Metrik *Root Mean Squared Error* (RMSE) GBT lebih rendah yang berarti prediksinya rata-rata lebih dekat pada harga asli. Nilai R² sebesar 0.7254 menandakan bahwa model ini dapat menjelaskan 72.5% ragam variasi dari penentuan tarif taksi.

#### 4.2 Analisis Tingkat Kepentingan Fitur (Feature Importance)
- **Proses:** Sistem menarik nilai `.featureImportances` bawaan algoritma yang memetakan bobot persentase dari setiap fitur terhadap target akhir.
- **Output:** Dihasilkan **Visualisasi Bar Chart**.
- **Pembahasan:** Dari hasil grafik, diketahui bahwa fitur jarak dan koordinat letak penurunan (*dropoff location*) menempati urutan paling krusial. Ini membuktikan bahwa dalam bisnis taksi, "seberapa jauh Anda pergi" adalah segalanya ketimbang faktor jumlah penumpang maupun tanggal.

#### 4.3 Optimasi Hyperparameter (Tuning)
- **Proses:** Menggunakan komponen PySpark `TrainValidationSplit` yang jauh lebih aman secara komputasi lokal daripada `CrossValidator`. Sistem membagi data dengan rasio 80% Latih dan 20% Validasi, lalu menguji variasi `maxDepth` [5, 8] serta `numTrees` [20, 30].
- **Output:** Komputer menemukan setelan paling prima pada **`maxDepth=8`** dan **`numTrees=30`**.
- **Pembahasan:** Hasil eksperimen komputasi membuktikan nilai *error* RMSE Random Forest berhasil **turun dari 2.2794 menjadi 2.2738**. Ini menandakan penyetelan (*tuning*) model yang dieksekusi membuahkan hasil positif.

#### 4.4 Eksekusi Prediksi Data Uji Akhir (Test Submission)
- **Proses:** Model final disuruh untuk "menebak" harga untuk sekumpulan data tes baru (`test.csv`) yang sama sekali belum memiliki tarif dasar.
- **Penanganan Bug Windows:** Mengingat PySpark rentan terkena *error environment* bawaan Winutils (`checkHadoopHomeInner`), output PySpark ini langsung kami konversi secara aman (di- *cast*) ke bentuk **Pandas DataFrame**.
- **Output:** Tergenerasinya file fisik **`predictions.csv`** dengan kolom `key` dan `fare_amount`. File bersih ini adalah pelengkap tugas untuk di-*submit* pada Kaggle.

---

### BAB V. KESIMPULAN

Berdasarkan keseluruhan proyek Big Data ini, dapat disimpulkan bahwa:
1. Pemrosesan data besar dapat dilakukan dengan mulus menggunakan ekosistem terdistribusi Apache Spark (PySpark DataFrame dan PySpark ML).
2. Dari segi keakuratan matematis, model arsitektur **Gradient Boosting** terbukti lebih mumpuni dari Random Forest dalam menangkap hubungan *non-linear* pada dataset tarif taksi NYC.
3. Fitur rekayasa kustom (pengukuran jarak absolut Euclidean) secara mutlak terbukti menjadi komponen paling vital yang memungkinkan model kami menebak harga argometer secara presisi.
