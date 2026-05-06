"""Halaman Tentang — info model, dataset, dan tech stack."""

import streamlit as st

# ─────────────────────────────────────────────────────────────────
# ⚠️  UPDATE METRIK INI sesuai hasil akhir dari Modelling_v4.ipynb
# ─────────────────────────────────────────────────────────────────
MODEL_F05   = "0.7415"   # F0.5 Score
MODEL_ACC   = "84.46%"   # Accuracy
MODEL_PREC  = "78.49%"   # Precision
MODEL_REC   = "60.73%"   # Recall
MODEL_AUC   = "0.9144"   # ROC-AUC (opsional)
THRESHOLD   = 0.6       # Custom threshold yang digunakan
N_FEATURES  = 31         # Jumlah fitur setelah engineering


def render() -> None:
    """Render halaman tentang."""
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown(
        "Hotel Booking Cancellation Predictor — aplikasi machine learning "
        "untuk memprediksi pembatalan booking hotel berbasis LightGBM."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Tentang Model")
        st.markdown(
            f"""
            - **Algoritma:** Stacking
            - **Versi Notebook:** Modelling_v4
            - **Pipeline:** ColumnTransformer (OHE + BinaryEncoder + TargetEncoder + RobustScaler)
            - **Dataset:** Hotel Bookings (119,390 baris, 2015–2017)
            - **Jumlah Fitur:** {N_FEATURES} fitur (raw + engineered)
            - **Target:** `is_canceled` (binary: 0 = tidak batal, 1 = batal)
            - **Threshold:** `{THRESHOLD}` (custom, dioptimasi F0.5)
            - **Class imbalance:** ditangani dalam pipeline
            """
        )

        st.markdown("### 📊 Performa Model")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("F0.5 Score", MODEL_F05)
        with m2:
            st.metric("Precision", MODEL_PREC)
        with m3:
            st.metric("Recall", MODEL_REC)
        m4, m5 = st.columns(2)
        with m4:
            st.metric("Accuracy", MODEL_ACC)
        with m5:
            st.metric("ROC-AUC", MODEL_AUC)


    with col2:
        st.markdown("### 📚 Sumber Dataset")
        st.markdown(
            """
            Dataset hotel booking dari Kaggle, berisi data dari 2 hotel
            di **Portugal** periode **2015–2017**:

            - **City Hotel** 
            - **Resort Hotel** 

            """
        )

        st.markdown("### 👨‍💻 Tech Stack")
        st.markdown(
            """
            - **Python 3.10+**, Streamlit
            - **Scikit-learn**, **Stacking**
            - **category-encoders** (BinaryEncoder, One Hot Encoder, TargetEncoder)
            - **SHAP** untuk explainability
            - **Plotly** untuk visualisasi interaktif
            - **Pandas**, **NumPy**
            """
        )

    st.markdown("---")

    st.markdown("### 🔍 Tentang SHAP (Explainability)")
    st.markdown(
        """
        Aplikasi menyediakan dua jenis penjelasan menggunakan SHAP (SHapley Additive exPlanations):

        - **SHAP Lokal** menjelaskan kenapa model memprediksi seperti itu untuk **booking spesifik**.
          Setiap fitur punya kontribusi positif (mendorong cancel) atau negatif (menahan cancel).
        - **SHAP Global** menjelaskan fitur mana yang **paling penting di mata model secara umum**.
          Dihitung dari rata-rata nilai absolut SHAP (`mean |SHAP|`) di seluruh data training.

        **Top 3 fitur paling berpengaruh (berdasarkan SHAP Global):**
        `country_grouped` → `agent` → `market_segment`
        """
    )

    st.markdown("---")

    st.info(
        "📝 **Catatan:** Model ini untuk tujuan pembelajaran dan eksplorasi. "
        "Untuk produksi, lakukan re-training berkala dengan data terbaru."
    )
    st.warning(
        "⚠️ Hasil prediksi adalah estimasi probabilistik. "
        "Kombinasikan dengan judgement manusia untuk keputusan bisnis."
    )
