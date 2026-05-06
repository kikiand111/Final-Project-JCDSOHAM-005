"""Halaman Beranda — intro singkat dan call-to-action ke Prediksi."""

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# ⚠️  UPDATE METRIK INI sesuai hasil akhir dari Modelling_v4.ipynb
# ─────────────────────────────────────────────────────────────────
MODEL_F05   = "0.7415"   # F0.5 Score
MODEL_ACC   = "84.46%"   # Accuracy
MODEL_PREC  = "78.49%"   # Precision
MODEL_REC   = "60.73%"   # Recall
THRESHOLD   = 0.6       # Custom threshold yang digunakan


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset hotel bookings dengan caching."""
    return pd.read_csv("hotel_bookings.csv")


def render() -> None:
    """Render halaman beranda."""
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown(
        "Tool prediksi pembatalan booking hotel berbasis machine learning. "
        "Mendukung prediksi tunggal (manual input) maupun batch (upload CSV)."
    )

    st.markdown("---")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("File `hotel_bookings.csv` tidak ditemukan di folder project.")
        return

    # ── METRIC CARDS ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Data Training", f"{len(df):,}")
    with c2:
        cancel_rate = df["is_canceled"].mean() * 100
        st.metric("Tingkat Pembatalan Historis", f"{cancel_rate:.1f}%")
    with c3:
        st.metric("F0.5 Score", MODEL_F05)
    with c4:
        st.metric("Threshold Keputusan", f"{THRESHOLD:.0%}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Untuk Apa Aplikasi Ini?")
        st.markdown(
            """
            Membantu pihak hotel mengidentifikasi booking yang berisiko
            tinggi dibatalkan, sehingga bisa:

            - Antisipasi potensi kerugian finansial
            - Optimasi room availability
            - Strategi marketing tepat sasaran
            - Identifikasi pola booking berisiko
            """
        )

    with col_b:
        st.markdown("### ⚡ Fitur Utama")
        st.markdown(
            """
            - **Prediksi tunggal** — input form 27 field, hasil + rekomendasi
            - **Prediksi batch** — upload CSV, prediksi banyak booking sekaligus
            - **SHAP Lokal** — pengaruh fitur untuk booking spesifik
            - **SHAP Global** — fitur paling berpengaruh di model umum
            - **Rekomendasi tindakan** — saran konkret per level risiko
            """
        )


    st.markdown("---")
    st.info(
        "💡 **Mulai sekarang:** klik menu **🔮 Prediksi** di sidebar kiri."
    )
