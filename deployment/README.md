# 🏨 Hotel Booking Cancellation Predictor

Aplikasi Streamlit untuk memprediksi pembatalan booking hotel.

## 📁 Struktur Project

```
hotel-prediction/
├── app.py                      # File utama Streamlit
├── style.css                   # Custom CSS styling
├── requirements.txt            # Dependencies
├── README.md                   # File ini
│
├── .streamlit/
│   └── config.toml             # Tema Streamlit
│
├── app_pages/                  # Modul halaman
│   ├── __init__.py
│   ├── home.py                 # Halaman beranda
│   ├── eda.py                  # Halaman EDA
│   ├── prediction.py           # Halaman prediksi
│   └── about.py                # Halaman tentang
│
├── hotel_bookings.csv          # Dataset (siapkan sendiri)
├── model.pkl                   # Model hasil training (siapkan sendiri)
└── scaler.pkl                  # Scaler preprocessing (opsional)
```

## 🚀 Cara Menjalankan

### 1. Setup virtual environment (rekomendasi)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pastikan file ini ada di folder project:

- `hotel_bookings.csv` (dataset)
- `model.pkl` (model hasil training, bisa ditambahkan nanti)

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## 🎨 Kustomisasi

- **Warna tema:** edit `.streamlit/config.toml` (primaryColor)
- **Style detail:** edit `style.css`
- **Tambah halaman:** buat file baru di `app_pages/`, tambahkan import dan routing di `app.py`

## 📦 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Pilih repo → set main file: `app.py`
5. Deploy → otomatis dapat URL publik

## ⚠️ Catatan

Halaman **Prediksi** saat ini menggunakan logika demo (placeholder).
Setelah model selesai training:

1. Simpan model: `joblib.dump(model, "model.pkl")`
2. Edit `app_pages/prediction.py` → uncomment bagian `load_model()`
3. Update fungsi `predict_cancellation()` dengan pipeline asli Anda
