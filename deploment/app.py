"""
Hotel Booking Cancellation Predictor
"""
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

import streamlit as st
import traceback
try:
    from app_pages import home, prediction, about
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_ERROR = traceback.format_exc()

# ============ KONFIGURASI HALAMAN ============
st.set_page_config(
    page_title="Hotel Booking Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============ LOAD CUSTOM CSS ============
def load_css(file_path: str) -> None:
    """Load file CSS untuk styling kustom."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS tidak ditemukan: {file_path}")


load_css("style.css")


# ============ SIDEBAR NAVIGASI ============
with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>🏨 Hotel Predictor</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sidebar-version'>v1.0</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Menu navigasi - hanya 3 item
    st.markdown("**NAVIGASI**")
    page = st.radio(
        label="Pilih halaman",
        options=["🏠 Beranda", "🔮 Prediksi", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Info model
    st.markdown("**INFO MODEL**")
    st.markdown(
        """
        <div class='info-card'>
            <div class='info-label'>Algoritma</div>
            <div class='info-value'>Stacking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='info-card success'>
            <div class='info-label'>F0.5 Score</div>
            <div class='info-value'>0.7415</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("© 2026 Hotel Predictor")


# ============ ROUTING HALAMAN ============
if IMPORT_ERROR:
    st.error("Import Error:")
    st.code(IMPORT_ERROR)
    st.stop()

if page == "🏠 Beranda":
    home.render()
elif page == "🔮 Prediksi":
    prediction.render()
elif page == "ℹ️ Tentang":
    about.render()