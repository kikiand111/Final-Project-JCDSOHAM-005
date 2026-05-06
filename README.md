# 🏨 Hotel Booking Cancellation Prediction

> Proyek analitik data dan machine learning end-to-end yang berfokus pada prediksi pembatalan pemesanan hotel untuk mendukung pengambilan keputusan berbasis data.

**Tim:** Ika · Kiki · Aulia

---

## 📌 Latar Belakang

Tingkat pembatalan pemesanan hotel yang tinggi menyebabkan kerugian pendapatan yang signifikan dan inefisiensi operasional. Proyek ini menggabungkan eksplorasi data interaktif melalui **Tableau** dengan sistem prediksi berbasis **Machine Learning** untuk membantu hotel mengidentifikasi pemesanan berisiko tinggi sejak dini dan mengambil tindakan preventif secara tepat waktu.

---

## 🎯 Tujuan Proyek

- Memahami pola dan faktor utama yang berkontribusi terhadap pembatalan pemesanan hotel
- Membangun model prediksi pembatalan dengan performa tinggi
- Menyajikan insight bisnis secara visual dan interaktif melalui dashboard
- Mendeploy model ke dalam aplikasi web yang dapat digunakan langsung oleh tim operasional

---

## 🗂️ Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **Periode** | 2015 – 2017 |
| **Jumlah Baris** | 119.390 |
| **Jumlah Kolom** | 32 |
| **Jenis Hotel** | City Hotel & Resort Hotel |
| **Target** | `is_canceled` (0 = Tidak Batal, 1 = Batal) |
| **Tingkat Pembatalan** | ~37% |

---

## 🔄 Alur Pengerjaan Proyek

> Lihat flowchart visual lengkap pada file `flowchart.png`.

```
1. Business & Data Understanding
         ↓
2. Exploratory Data Analysis (EDA)
         ↓
3. Feature Engineering & Data Cleaning
         ↓
4. Modelling (Eksperimen → Tuning → Final Model)
         ↓
5. Evaluasi & Interpretasi (Feature Importance via LightGBM)
         ↓
6. Deployment
    ├── Streamlit App (Prediksi Interaktif)
    └── Tableau Dashboard (Insight Bisnis)
```

---

## 📋 Tahapan Pengerjaan

### 1. Business & Data Understanding

- **Masalah Bisnis:** Tingginya tingkat pembatalan menyebabkan kebocoran pendapatan dan inefisiensi operasional
- **KPI Utama:** Cancel Rate, Lead Time, Average Daily Rate (ADR), dan RevPAR
- **Sumber Data:** Hotel Booking Demand Dataset — dataset publik yang mencakup City Hotel dan Resort Hotel tahun 2015–2017
- **Pemangku Kepentingan:** Manajemen Hotel, Revenue Manager, dan Tim Reservasi

### 2. Exploratory Data Analysis (EDA)

- Penanganan missing values, outlier, dan inkonsistensi tipe data
- Analisis distribusi target (37% pembatalan), univariat, dan bivariat
- Identifikasi korelasi antar fitur dan potensi target leakage
- Visualisasi pola pembatalan berdasarkan segmen, tipe hotel, dan periode waktu

### 3. Feature Engineering & Data Cleaning

- `country` dikelompokkan menjadi `country_grouped` berdasarkan analisis threshold
- Binning fitur zero-inflated ke dalam kategori bermakna (mis. `prev_cancel_bin`, `booking_changes_bin`, `parking_bin`)
- `arrival_month` ditransformasi menjadi `arrival_season` dan komponen siklus `month_sin` / `month_cos`
- Konstruksi fitur interaksi, contoh: `lead_x_transient`
- Outlier ADR ekstrem diidentifikasi dan dihapus melalui cross-check kolom terkait
- `deposit_type` dikecualikan dari model karena terindikasi **target leakage**
- Dataset original (`df`) dipisah dari dataset siap model (`df_model`); setiap langkah cleaning dicatat via `log_step()`

### 4. Modelling

**Pipeline preprocessing:** OHE · BinaryEncoder · TargetEncoder · Log + RobustScaler (via ColumnTransformer)

**Penanganan imbalance:** `class_weight='balanced'` diterapkan pada seluruh model yang mendukung parameter tersebut

**Eksperimen model** (evaluasi menggunakan F0.5 Score — Stratified K-Fold CV):

| Rank | Model | F0.5 Mean | F0.5 Std |
|---|---|---|---|
| 🥇 1 | **Stacking** | **0.7241** | 0.0052 |
| 🥈 2 | Voting | 0.7232 | 0.0043 |
| 3 | CatBoost | 0.7177 | 0.0049 |
| 4 | LightGBM | 0.7146 | 0.0047 |
| 5 | XGBoost | 0.7134 | 0.0047 |
| 6 | RandomForest | 0.7066 | 0.0032 |
| 7 | LogisticRegression | 0.6406 | 0.0031 |

**Model final:** Stacking Classifier — disimpan sebagai `final_model_v2.pkl`

> Interpretabilitas model (SHAP & Feature Importance) menggunakan **LightGBM sebagai proxy**, karena LightGBM merupakan base estimator terkuat dalam ensemble Stacking.

### 5. Evaluasi & Interpretasi

- Metrik evaluasi: Precision, Recall, F1, **F0.5** (metrik utama), AUC-ROC, AUC-PR
- Threshold dioptimasi berdasarkan **cost-sensitive analysis** → threshold = 0.6
- Feature Importance berbasis **Gain** dari LightGBM (proxy Stacking)
- Analisis SHAP untuk interpretabilitas prediksi per kasus

---

## 📈 Hasil Evaluasi Model

### Metrik Performa — Stacking | Threshold = 0.6

| Metrik | Nilai |
|---|---|
| **AUC-ROC** | **0.9144** |
| **AUC-PR** | **0.8043** |
| Accuracy | 0.8446 |
| Precision (Cancel) | 0.7849 |
| Recall (Cancel) | 0.6073 |
| F1-Score (Cancel) | 0.6848 |
| **F0.5-Score (Cancel)** | **0.7415** |

> Model diprioritaskan pada **Precision tinggi (F0.5)** — tindakan retensi hanya diarahkan ke tamu yang benar-benar berisiko batal agar biaya intervensi tidak terbuang sia-sia.

### Confusion Matrix

|  | Predicted Not Cancel | Predicted Cancel |
|---|---|---|
| **Actual Not Cancel** | TN: 17.425 (67.6%) | FP: 1.192 (4.6%) |
| **Actual Cancel** | FN: 2.813 (10.9%) | TP: 4.350 (16.9%) |

> **FP/TP ratio = 0.27** — setiap 1 prediksi cancel yang tepat, hanya terdapat 0.27 false alarm.

### Feature Importance — Gain (Top 10) via LightGBM

| Rank | Fitur | % of Total Gain |
|---|---|---|
| 1 | `agent` | 12.4% |
| 2 | `lead_x_transient` | 10.6% |
| 3 | `country_grouped` | 10.2% |
| 4 | `lead_time` | 10.1% |
| 5 | `adr` | 7.0% |
| 6 | `parking_bin_has_parking` | 5.0% |
| 7 | `market_segment_2` | 4.2% |
| 8 | `total_of_special_requests` | 3.8% |
| 9 | `market_segment_1` | 3.6% |
| 10 | `no_commitment` | 3.6% |

---

## 💰 Dampak Bisnis — Cost-Sensitive Analysis

Threshold 0.6 dipilih berdasarkan analisis biaya asimetris antara dua jenis kesalahan prediksi.

| Jenis Error | Implikasi Bisnis | Total Biaya |
|---|---|---|
| **False Positive** | Retention cost sia-sia (walking guest) | €834.400 |
| **False Negative** | Kamar kosong tidak terantisipasi (opportunity cost) | €843.900 |

### Perbandingan Total Biaya

| Skenario | Total Biaya |
|---|---|
| Tanpa Model (Baseline) | €2.148.900 |
| **Dengan Model (t = 0.6)** | **€1.678.300** |
| **Penghematan** | **€470.600 (↓ 21.9%)** |

---

## 🔗 Deliverables

| Deliverable | Link |
|---|---|
| 📊 Tableau Dashboard | [Hotel Booking Cancellation Dashboard](https://public.tableau.com/app/profile/ika.christine.purba/viz/hotelbookingcancellation_17775939354600/Dash_BusinesOverview) |
| 🌐 Streamlit App & Final Model | [Google Drive](https://drive.google.com/drive/folders/1Lr5ZKjMFGcHcFzKKWeVOwBz_cJL1BQ3L?usp=sharing) |
| 📦 Dataset | Hotel Booking Demand Dataset (Kaggle — Publik) |

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| **Bahasa** | Python 3.10+ |
| **ML & Preprocessing** | Stacking, LightGBM, CatBoost, XGBoost, scikit-learn, category-encoders |
| **Tuning** | Optuna |
| **Interpretabilitas** | SHAP, Feature Importance (LightGBM proxy) |
| **Aplikasi Web** | Streamlit |
| **Visualisasi** | Tableau, Seaborn, Matplotlib |
| **Notebook** | Jupyter Notebook / VS Code |
| **Version Control** | Git + GitHub |

---

## 👥 Tim Proyek

Ika · Kiki · Aulia

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan portofolio guna mendemonstrasikan kemampuan di bidang Business Intelligence dan Data Science.

Dataset bersumber dari publikasi terbuka: *Antonio, Almedia, and Nunes (2019).*
