"""Halaman Prediksi — single booking & batch CSV prediction."""

import math
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────────
# Model berada satu level di atas folder app_pages
MODEL_PATH = Path(__file__).resolve().parent.parent / "final_model_v5.pkl"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_TO_NUM: dict[str, int] = {m: i + 1 for i, m in enumerate(MONTHS)}

# Sinkron dengan training (Section 3.1 Modelling_v4.ipynb)
SEASON_MAP: dict[int, str] = {
    12: "Winter", 1: "Winter",  2: "Winter",
     3: "Spring", 4: "Spring",  5: "Spring",
     6: "Summer", 7: "Summer",  8: "Summer",
     9: "Fall",  10: "Fall",   11: "Fall",
}

# country_grouped: top 36 negara + "Other" (Section 3.2 Modelling_v4.ipynb)
COUNTRIES = [
    "AGO", "ARG", "AUS", "AUT", "BEL", "BRA", "CHE", "CHN",
    "CZE", "DEU", "DNK", "DZA", "ESP", "FIN", "FRA", "GBR",
    "GRC", "HRV", "HUN", "IND", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LUX", "MAR", "NLD", "NOR", "Other", "POL", "PRT",
    "ROU", "RUS", "SWE", "TUR", "USA",
]

# ─────────────────────────────────────────────────────────────────
# GLOBAL SHAP IMPORTANCE
# Sumber: Section 9.5 Modelling_v4.ipynb — Mean |SHAP| dari LightGBM
# pada 2000 sampel X_test.
#
# Nilai yang terkonfirmasi dari teks notebook:
#   country_grouped          = 0.9272  ← dari Kesimpulan Bisnis
#   parking_bin_has_parking  = 0.5478  ← dari Kesimpulan Bisnis
#   lead_x_transient         = 0.3941  ← dari Kesimpulan Bisnis
#
# Nilai lainnya: estimasi berdasarkan urutan beeswarm (top-15).
# Update dengan output cell 9.5 setelah re-training.
# ─────────────────────────────────────────────────────────────────
GLOBAL_IMPORTANCE: dict[str, float] = {
    "country_grouped":          0.9272,   # ✓ confirmed
    "agent":                    0.6850,   # ~ estimated (rank 2 di beeswarm)
    "parking_bin_has_parking":  0.5478,   # ✓ confirmed
    "parking_bin_no_parking":   0.4150,   # ~ estimated (rank 4 di beeswarm)
    "lead_x_transient":         0.3941,   # ✓ confirmed
    "no_commitment":            0.2650,   # ~ estimated (rank 6)
    "lead_time":                0.2180,   # ~ estimated (rank 7)
    "prev_cancel_bin_never":    0.1740,   # ~ estimated (rank 8)
    "booking_changes_bin_none": 0.1310,   # ~ estimated (rank 9)
    "prev_loyal_bin_none":      0.0920,   # ~ estimated (rank 10)
}

# ─────────────────────────────────────────────────────────────────
# THRESHOLD — sinkron dengan final_model_v5.pkl (Section 8.7)
# ─────────────────────────────────────────────────────────────────
THRESHOLD: float = 0.60


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan: `{MODEL_PATH}`")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model_dict = pickle.load(f)
        # final_model_v5.pkl adalah dict — ambil pipeline-nya
        if isinstance(model_dict, dict):
            return model_dict["pipeline"]
        return model_dict
    except Exception:
        st.error("Gagal memuat model.")
        st.code(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────
# HELPER: sin-cos encoding bulan
# ─────────────────────────────────────────────────────────────────
def _month_sincos(month_num: int) -> tuple[float, float]:
    a = 2 * math.pi * month_num / 12
    return math.sin(a), math.cos(a)


# ─────────────────────────────────────────────────────────────────
# BUILD FEATURES — single prediction (29 kolom, urutan = pipeline)
# FIX: no_commitment = (parking == no_parking) AND (special_req == 0)
#      Sinkron dengan Section 3.4 Modelling_v4.ipynb
# ─────────────────────────────────────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    month_num           = MONTH_TO_NUM[raw["arrival_date_month"]]
    month_sin, month_cos = _month_sincos(month_num)
    season              = SEASON_MAP[month_num]

    lead    = int(raw["lead_time"])
    ctype   = raw["customer_type"]
    lead_x_t = lead * (1 if ctype == "Transient" else 0)

    stays_wk  = int(raw["stays_in_week_nights"])
    stays_we  = int(raw["stays_in_weekend_nights"])
    is_wknd   = int(stays_wk == 0 and stays_we > 0)

    country = raw.get("country_grouped", "Other")
    if country not in COUNTRIES:
        country = "Other"

    # ✅ FIX: no_commitment = tanpa parkir DAN tanpa special request
    #         (bukan dari deposit_type — deposit_type adalah leaky feature)
    parking_bin = raw.get("parking_bin", "no_parking")
    no_commit   = int(
        parking_bin == "no_parking"
        and int(raw.get("total_of_special_requests", 0)) == 0
    )

    # company: 0 = bukan korporat → NaN; >0 = ID perusahaan korporat
    company_input = raw.get("company", 0)
    company_val   = float("nan") if (not company_input or company_input == 0) else float(company_input)

    row = {
        "hotel":                     raw["hotel"],
        "lead_time":                 lead,
        "stays_in_weekend_nights":   stays_we,
        "stays_in_week_nights":      stays_wk,
        "adults":                    int(raw["adults"]),
        "meal":                      raw["meal"],
        "market_segment":            raw["market_segment"],
        "distribution_channel":      raw["distribution_channel"],
        "is_repeated_guest":         int(raw["is_repeated_guest"]),
        "reserved_room_type":        raw["reserved_room_type"],
        "agent":                     raw["agent"],
        "company":                   company_val,
        "days_in_waiting_list":      int(raw["days_in_waiting_list"]),
        "customer_type":             ctype,
        "adr":                       float(raw["adr"]),
        "total_of_special_requests": int(raw["total_of_special_requests"]),
        "country_grouped":           country,
        "arrival_season":            season,
        "month_sin":                 month_sin,
        "month_cos":                 month_cos,
        "babies_bin":                raw["babies_bin"],
        "children_bin":              raw["children_bin"],
        "parking_bin":               parking_bin,
        "prev_cancel_bin":           raw["prev_cancel_bin"],
        "prev_loyal_bin":            raw["prev_loyal_bin"],
        "booking_changes_bin":       raw["booking_changes_bin"],
        "no_commitment":             no_commit,
        "lead_x_transient":          lead_x_t,
        "is_weekend_only":           is_wknd,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────
# BATCH FEATURE ENGINEERING
# FIX: no_commitment dari parking + special_requests (bukan deposit_type)
# FIX: prev_loyal_bin — some=1–3, many=≥4 (sinkron training Section 3.3.b)
# ─────────────────────────────────────────────────────────────────
def _prev_cancel_bin(n) -> str:
    n = int(n) if pd.notna(n) else 0
    return "never" if n == 0 else ("once" if n == 1 else "multiple")


def _prev_loyal_bin(n) -> str:
    # ✅ FIX: threshold sesuai notebook — some=1–3, many=≥4
    n = int(n) if pd.notna(n) else 0
    return "none" if n == 0 else ("some" if n <= 3 else "many")


def _booking_changes_bin(n) -> str:
    n = int(n) if pd.notna(n) else 0
    return "none" if n == 0 else ("some" if n <= 2 else "many")


def _batch_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temporal features
    month_nums       = df["arrival_date_month"].map(MONTH_TO_NUM).fillna(1).astype(int)
    df["month_sin"]  = month_nums.apply(lambda m: math.sin(2 * math.pi * m / 12))
    df["month_cos"]  = month_nums.apply(lambda m: math.cos(2 * math.pi * m / 12))
    df["arrival_season"] = month_nums.map(SEASON_MAP)

    # country_grouped
    if "country" in df.columns and "country_grouped" not in df.columns:
        df["country_grouped"] = df["country"].apply(
            lambda c: c if c in COUNTRIES else "Other"
        )

    # Interaction feature
    df["lead_x_transient"] = df["lead_time"].fillna(0) * (
        df["customer_type"].eq("Transient").astype(int)
    )

    # Weekend-only flag
    df["is_weekend_only"] = (
        (df["stays_in_week_nights"].fillna(0) == 0) &
        (df["stays_in_weekend_nights"].fillna(0) > 0)
    ).astype(int)

    # Bin features
    df["babies_bin"]  = df["babies"].fillna(0).apply(
        lambda x: "has_baby" if int(x) > 0 else "no_baby")
    df["children_bin"] = df["children"].fillna(0).apply(
        lambda x: "has_children" if int(x) > 0 else "no_children")
    df["parking_bin"] = df["required_car_parking_spaces"].fillna(0).apply(
        lambda x: "has_parking" if int(x) > 0 else "no_parking")
    df["prev_cancel_bin"]     = df["previous_cancellations"].fillna(0).apply(_prev_cancel_bin)
    df["prev_loyal_bin"]      = df["previous_bookings_not_canceled"].fillna(0).apply(_prev_loyal_bin)
    df["booking_changes_bin"] = df["booking_changes"].fillna(0).apply(_booking_changes_bin)

    # ✅ FIX: no_commitment = tanpa parkir DAN tanpa special request
    df["no_commitment"] = (
        (df["required_car_parking_spaces"].fillna(0) == 0) &
        (df["total_of_special_requests"].fillna(0) == 0)
    ).astype(int)

    # company: 0 → NaN; nilai asli dipertahankan jika ada
    if "company" in df.columns:
        df["company"] = pd.to_numeric(df["company"], errors="coerce")
        df.loc[df["company"] == 0, "company"] = float("nan")
    else:
        df["company"] = float("nan")

    # agent: 0 → NaN (tidak melalui agen)
    if "agent" in df.columns:
        df["agent"] = pd.to_numeric(df["agent"], errors="coerce")
        df.loc[df["agent"] == 0, "agent"] = float("nan")

    COLS = [
        "hotel", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "meal", "market_segment", "distribution_channel",
        "is_repeated_guest", "reserved_room_type", "agent", "company",
        "days_in_waiting_list", "customer_type", "adr", "total_of_special_requests",
        "country_grouped", "arrival_season", "month_sin", "month_cos",
        "babies_bin", "children_bin", "parking_bin",
        "prev_cancel_bin", "prev_loyal_bin", "booking_changes_bin",
        "no_commitment", "lead_x_transient", "is_weekend_only",
    ]
    for c in COLS:
        if c not in df.columns:
            df[c] = 0
    return df[COLS]


# ─────────────────────────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────────────────────────
def _gauge_chart(prob: float) -> go.Figure:
    color = "#E74C3C" if prob >= 0.65 else ("#F39C12" if prob >= 0.40 else "#27AE60")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,   40], "color": "#D5F5E3"},
                {"range": [40,  65], "color": "#FDEBD0"},
                {"range": [65, 100], "color": "#FADBD8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        },
        title={"text": "Probabilitas Pembatalan"},
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def _shap_local_chart(shap_vals, feat_names):
    pairs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:12]
    names  = [p[0] for p in pairs][::-1]
    vals   = [p[1] for p in pairs][::-1]
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Lokal — Pengaruh Fitur pada Prediksi Ini",
        xaxis_title="SHAP Value",
        height=420, margin=dict(t=50, b=20, l=170, r=80),
    )
    return fig


def _shap_global_chart():
    # Urutkan ascending agar bar terbesar di atas
    items = sorted(GLOBAL_IMPORTANCE.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[i[1] for i in items],
        y=[i[0] for i in items],
        orientation="h",
        marker_color="#7C5CBF",
        text=[f"{v:.4f}" for v in [i[1] for i in items]],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Global — Mean |SHAP| dari LightGBM (2000 sampel X_test)",
        xaxis_title="Mean |SHAP Value|",
        height=420, margin=dict(t=50, b=20, l=210, r=100),
    )
    return fig


def _recommendation(prob: float, pred: int) -> None:
    if pred == 1:
        if prob >= 0.75:
            st.error("🔴 **RISIKO SANGAT TINGGI**")
            st.markdown("""
            - Hubungi tamu untuk konfirmasi ulang segera
            - Tawarkan insentif retensi (upgrade kamar, F&B voucher)
            - Aktifkan strategi overbooking / re-sell kamar
            """)
        else:
            st.warning("🟡 **RISIKO TINGGI**")
            st.markdown("""
            - Monitor booking lebih ketat
            - Kirim reminder konfirmasi 48 jam sebelum check-in
            - Siapkan opsi pengganti dari waitlist
            """)
    else:
        if prob >= 0.40:
            st.info("🟠 **RISIKO MENENGAH** — Pantau, kirim konfirmasi standar.")
        else:
            st.success("🟢 **RISIKO RENDAH** — Booking terlihat stabil.")


# ─────────────────────────────────────────────────────────────────
# RENDER UTAMA
# ─────────────────────────────────────────────────────────────────
def render() -> None:
    st.title("🔮 Prediksi Pembatalan Booking")

    model = load_model()
    if model is None:
        return

    tab_single, tab_batch, tab_global = st.tabs(
        ["📋 Prediksi Tunggal", "📂 Prediksi Batch (CSV)", "📊 SHAP Global"]
    )

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — PREDIKSI TUNGGAL
    # ══════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown("Isi form berikut lalu klik **Prediksi**.")

        with st.form("form_single"):

            st.subheader("📅 Informasi Booking")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                hotel = st.selectbox("Tipe Hotel", ["City Hotel", "Resort Hotel"])
            with c2:
                arrival_month = st.selectbox("Bulan Kedatangan", MONTHS, index=6)
            with c3:
                lead_time = st.number_input(
                    "Lead Time (hari)", min_value=0, max_value=709, value=30, step=1,
                    help="Hari antara pemesanan dan check-in. (0–709)"
                )
            with c4:
                days_waiting = st.number_input(
                    "Hari di Waiting List", min_value=0, max_value=180, value=0, step=1,
                    help="Hari booking di waiting list. (0–180)"
                )

            st.subheader("🛏️ Lama Menginap")
            c1, c2, c3 = st.columns(3)
            with c1:
                stays_weekend = st.number_input(
                    "Malam Akhir Pekan", min_value=0, max_value=10, value=1, step=1,
                    help="Sabtu/Minggu. (0–10)"
                )
            with c2:
                stays_week = st.number_input(
                    "Malam Hari Kerja", min_value=0, max_value=21, value=2, step=1,
                    help="Senin–Jumat. (0–21)"
                )
            with c3:
                adults = st.number_input(
                    "Dewasa", min_value=1, max_value=10, value=2, step=1,
                    help="Jumlah tamu dewasa. (1–10)"
                )

            st.subheader("👶 Tamu Tambahan")
            c1, c2 = st.columns(2)
            with c1:
                babies_bin = st.selectbox(
                    "Bayi",
                    options=["no_baby", "has_baby"],
                    format_func=lambda x: "Tidak ada bayi" if x == "no_baby" else "Ada bayi (≥1)",
                )
            with c2:
                children_bin = st.selectbox(
                    "Anak-anak",
                    options=["no_children", "has_children"],
                    format_func=lambda x: "Tidak ada anak-anak" if x == "no_children" else "Ada anak-anak (≥1)",
                )

            st.subheader("🚪 Kamar & Tarif")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                reserved_room = st.selectbox(
                    "Tipe Kamar Dipesan",
                    ["A", "B", "C", "D", "E", "F", "G", "H", "L"]
                )
            with c2:
                meal = st.selectbox(
                    "Paket Makan", ["BB", "HB", "FB", "SC"],
                    help="BB=Bed & Breakfast, HB=Half Board, FB=Full Board, SC=Self Catering"
                )
            with c3:
                adr = st.number_input(
                    "ADR (€/malam)", min_value=0.0, max_value=510.0, value=100.0, step=1.0,
                    help="Average Daily Rate. (0–510)"
                )
            with c4:
                total_special = st.number_input(
                    "Permintaan Khusus", min_value=0, max_value=5, value=0, step=1,
                    help="Jumlah permintaan khusus. (0–5)"
                )

            st.subheader("🚗 Parkir")
            # ✅ parking_bin langsung dikumpulkan di sini — dipakai untuk
            #    hitung no_commitment = (no_parking) AND (special_req == 0)
            c1, c2 = st.columns(2)
            with c1:
                parking_bin = st.selectbox(
                    "Parkir",
                    options=["no_parking", "has_parking"],
                    format_func=lambda x: "Tidak butuh parkir" if x == "no_parking" else "Butuh parkir",
                    help="Tamu yang meminta parkir hampir tidak pernah cancel (cancel rate ≈0%)"
                )
            with c2:
                # Tampilkan kalkulasi no_commitment secara real-time (informasi saja)
                st.caption("ℹ️ `no_commitment` = tanpa parkir **DAN** tanpa permintaan khusus")

            st.subheader("🌐 Sumber Booking")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                market_segment = st.selectbox(
                    "Segmen Pasar",
                    ["Online TA", "Offline TA/TO", "Direct", "Corporate",
                     "Groups", "Complementary", "Aviation"]
                )
            with c2:
                distribution_channel = st.selectbox(
                    "Channel Distribusi",
                    ["TA/TO", "Direct", "Corporate", "GDS"]
                )
            with c3:
                customer_type = st.selectbox(
                    "Tipe Pelanggan",
                    ["Transient", "Transient-Party", "Contract", "Group"]
                )
            with c4:
                country_grouped = st.selectbox(
                    "Negara Asal Tamu", COUNTRIES,
                    index=COUNTRIES.index("PRT")
                )

            st.subheader("🤝 Agen & Perusahaan")
            c1, c2 = st.columns(2)
            with c1:
                agent_input = st.number_input(
                    "ID Agen (0 = tanpa agen)",
                    min_value=0, max_value=535, value=0, step=1,
                    help="Kode numerik agen. 0 = tidak melalui agen. (0–535)"
                )
                agent_val = float(agent_input) if agent_input > 0 else float("nan")
            with c2:
                company_input = st.number_input(
                    "ID Perusahaan / Korporat (0 = bukan korporat)",
                    min_value=0, max_value=543, value=0, step=1,
                    help="Kode numerik perusahaan korporat. 0 = booking individu. (0–543)"
                )
                company_val = float(company_input) if company_input > 0 else float("nan")

            st.subheader("📋 Riwayat Tamu")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                is_repeated = st.selectbox("Tamu Berulang?", ["Tidak (0)", "Ya (1)"])
                is_repeated_val = 1 if is_repeated.startswith("Ya") else 0
            with c2:
                prev_cancel_bin = st.selectbox(
                    "Pembatalan Sebelumnya",
                    options=["never", "once", "multiple"],
                    format_func=lambda x: {
                        "never":    "Tidak pernah (0)",
                        "once":     "Satu kali (1)",
                        "multiple": "Lebih dari satu (≥2)",
                    }[x],
                )
            with c3:
                prev_loyal_bin = st.selectbox(
                    "Booking Sukses Sebelumnya",
                    options=["none", "some", "many"],
                    format_func=lambda x: {
                        "none": "Tidak ada (0)",
                        "some": "Sedikit (1–3)",   # ✅ FIX: was 1–4
                        "many": "Banyak (≥4)",      # ✅ FIX: was ≥5
                    }[x],
                )
            with c4:
                booking_changes_bin = st.selectbox(
                    "Perubahan Booking",
                    options=["none", "some", "many"],
                    format_func=lambda x: {
                        "none": "Tidak ada (0)",
                        "some": "Sedikit (1–2)",
                        "many": "Banyak (≥3)",
                    }[x],
                )

            submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # PROSES SETELAH SUBMIT
        # ══════════════════════════════════════════════════════════
        if submitted:
            raw = {
                "hotel":                     hotel,
                "lead_time":                 lead_time,
                "stays_in_weekend_nights":   stays_weekend,
                "stays_in_week_nights":      stays_week,
                "adults":                    adults,
                "meal":                      meal,
                "market_segment":            market_segment,
                "distribution_channel":      distribution_channel,
                "is_repeated_guest":         is_repeated_val,
                "reserved_room_type":        reserved_room,
                "agent":                     agent_val,
                "company":                   company_val,
                "days_in_waiting_list":      days_waiting,
                "customer_type":             customer_type,
                "adr":                       adr,
                "total_of_special_requests": total_special,
                "country_grouped":           country_grouped,
                "arrival_date_month":        arrival_month,
                "parking_bin":               parking_bin,
                "babies_bin":                babies_bin,
                "children_bin":              children_bin,
                "prev_cancel_bin":           prev_cancel_bin,
                "prev_loyal_bin":            prev_loyal_bin,
                "booking_changes_bin":       booking_changes_bin,
            }

            # ── PREDIKSI ──
            try:
                X    = build_features(raw)
                prob = float(model.predict_proba(X)[0, 1])
                pred = int(prob >= THRESHOLD)

                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")

                col_g, col_r = st.columns([1.2, 1])
                with col_g:
                    st.plotly_chart(_gauge_chart(prob), use_container_width=True, key="gauge_single")
                with col_r:
                    st.markdown(f"**Probabilitas Pembatalan:** `{prob:.1%}`")
                    st.markdown(f"**Threshold:** `{THRESHOLD:.0%}`")
                    label = "🔴 **AKAN DIBATALKAN**" if pred == 1 else "🟢 **TIDAK DIBATALKAN**"
                    st.markdown(f"**Prediksi:** {label}")

                    season_disp = SEASON_MAP[MONTH_TO_NUM[arrival_month]]
                    no_commit   = int(parking_bin == "no_parking" and total_special == 0)
                    nc_label    = "⚠️ Ya" if no_commit else "Tidak"
                    we_label    = "Ya" if stays_week == 0 and stays_weekend > 0 else "Tidak"

                    st.caption(
                        f"Musim: **{season_disp}** · No-commitment: **{nc_label}** · "
                        f"Weekend only: **{we_label}**"
                    )

                st.markdown("---")
                _recommendation(prob, pred)

            except Exception:
                st.error("Terjadi kesalahan saat prediksi.")
                st.code(traceback.format_exc())

            # ── SHAP LOKAL + GLOBAL (berdampingan) ──
            with st.expander("🔍 SHAP — Pengaruh Fitur (Lokal & Global)", expanded=False):
                try:
                    import lightgbm as lgb

                    # Cari LightGBM dari dalam stacking
                    stacking = None
                    for _, step in model.steps:
                        if hasattr(step, "estimators_"):
                            stacking = step
                            break

                    lgbm_clf = None
                    if stacking is not None:
                        for est in stacking.estimators_:
                            if hasattr(est, "named_steps"):
                                for _, obj in est.named_steps.items():
                                    if isinstance(obj, lgb.LGBMClassifier):
                                        lgbm_clf = obj
                                        break
                            elif isinstance(est, lgb.LGBMClassifier):
                                lgbm_clf = est
                            if lgbm_clf is not None:
                                break

                    if lgbm_clf is None:
                        st.warning("LightGBM base estimator tidak ditemukan di dalam model.")
                    else:
                        # Preprocessing
                        ct_shap = model.named_steps["preprocessor"]
                        X_t = ct_shap.transform(X)

                        def _get_feature_names(ct):
                            names = []
                            for tname, trans, cols in ct.transformers_:
                                if tname == "remainder":
                                    continue
                                if hasattr(trans, "get_feature_names_out"):
                                    try:
                                        names.extend(trans.get_feature_names_out())
                                    except Exception:
                                        names.extend(cols if isinstance(cols, list) else [cols])
                                else:
                                    names.extend(cols if isinstance(cols, list) else [cols])
                            return names

                        feat_names = _get_feature_names(ct_shap)
                        feat_names = [
                            f.replace("country_grouped_", "Country: ")
                             .replace("market_segment_",  "Segment: ")
                             .replace("distribution_channel_", "Channel: ")
                            for f in feat_names
                        ]

                        if hasattr(X_t, "toarray"):
                            X_t = X_t.toarray()

                        explainer = shap.TreeExplainer(lgbm_clf)
                        sv = explainer.shap_values(X_t)
                        if isinstance(sv, list):
                            sv = sv[1]
                        sv = sv[0] if sv.ndim == 2 else sv

                        # ── Dua chart berdampingan ──────────────────
                        col_local, col_global = st.columns(2)
                        with col_local:
                            st.plotly_chart(
                                _shap_local_chart(sv, feat_names),
                                use_container_width=True,
                                key="shap_local_expander",
                            )
                        with col_global:
                            st.plotly_chart(
                                _shap_global_chart(),
                                use_container_width=True,
                                key="shap_global_expander",
                            )

                        # ── Penjelasan Model — Top 3 cards + tabel semua fitur ──
                        st.markdown("---")
                        st.markdown("#### 📋 Penjelasan Model — Seluruh Fitur Booking Ini")

                        # Top 3 cards
                        sorted_idx = np.argsort(np.abs(sv))[::-1]
                        top3_idx   = sorted_idx[:3]

                        CARD_CSS = """
                        <style>
                        .shap-card {
                            background: #F8F7FF;
                            border-radius: 10px;
                            padding: 14px 16px;
                            border-left: 4px solid #ccc;
                            height: 100%;
                        }
                        .shap-rank { font-size: 11px; color: #888; font-weight: 600;
                                     text-transform: uppercase; letter-spacing: .5px; }
                        .shap-feat { font-size: 14px; font-weight: 700; color: #2C2C2A;
                                     margin: 4px 0 2px; word-break: break-word; }
                        .shap-dir  { font-size: 12px; font-weight: 600; }
                        .shap-val  { font-size: 20px; font-weight: 700; margin-top: 6px; }
                        .up   { color: #E74C3C; }
                        .down { color: #2980B9; }
                        </style>
                        """
                        st.markdown(CARD_CSS, unsafe_allow_html=True)

                        def _render_card(col, rank, feat, val):
                            is_up  = val > 0
                            border = "#E74C3C" if is_up else "#2980B9"
                            arrow  = "▲ Meningkatkan cancel" if is_up else "▼ Menurunkan cancel"
                            cls    = "up" if is_up else "down"
                            with col:
                                st.markdown(
                                    f"""
                                    <div class="shap-card" style="border-left-color:{border}">
                                        <div class="shap-rank">#{rank} · Fitur Terpenting</div>
                                        <div class="shap-feat">{feat}</div>
                                        <div class="shap-dir {cls}">{arrow}</div>
                                        <div class="shap-val {cls}">{val:+.3f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        cols3 = st.columns(3)
                        for rank, i in enumerate(top3_idx, start=1):
                            _render_card(cols3[rank - 1], rank, feat_names[i], sv[i])

                        st.markdown("<div style='margin-top:16px'></div>",
                                    unsafe_allow_html=True)

                        # Tabel semua fitur — hanya yang |SHAP| > 0
                        all_rows = [
                            {
                                "Rank":    rank + 1,
                                "Fitur":   feat_names[i],
                                "SHAP":    round(float(sv[i]), 4),
                                "|SHAP|":  round(abs(float(sv[i])), 4),
                                "Arah":    "▲ Meningkatkan" if sv[i] > 0 else "▼ Menurunkan",
                            }
                            for rank, i in enumerate(sorted_idx)
                            if abs(sv[i]) > 0.0001
                        ]

                        if all_rows:
                            import pandas as pd

                            df_shap = pd.DataFrame(all_rows)

                            def _color_arah(val):
                                if "Meningkatkan" in str(val):
                                    return "color: #E74C3C; font-weight: 600"
                                return "color: #2980B9; font-weight: 600"

                            def _color_shap(val):
                                if val > 0:
                                    intensity = min(int(abs(val) / (abs(sv).max() + 1e-9) * 180), 180)
                                    return f"background-color: rgba(231,76,60,{intensity/255:.2f}); color: white"
                                elif val < 0:
                                    intensity = min(int(abs(val) / (abs(sv).max() + 1e-9) * 180), 180)
                                    return f"background-color: rgba(41,128,185,{intensity/255:.2f}); color: white"
                                return ""

                            st.markdown(
                                f"**{len(all_rows)} fitur** berkontribusi pada prediksi ini "
                                f"(diurutkan dari pengaruh terbesar)"
                            )
                            st.dataframe(
                                df_shap.style
                                .map(_color_arah, subset=["Arah"])
                                .map(_color_shap, subset=["SHAP"])
                                .format({"SHAP": "{:+.4f}", "|SHAP|": "{:.4f}"}),
                                use_container_width=True,
                                hide_index=True,
                                height=min(400, 35 * len(all_rows) + 38),
                            )

                except Exception:
                    st.warning("SHAP gagal dimuat.")
                    st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — BATCH
    # ══════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("""
        Upload CSV dengan kolom minimal:
        `hotel`, `lead_time`, `arrival_date_month`, `stays_in_weekend_nights`,
        `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`,
        `market_segment`, `distribution_channel`, `is_repeated_guest`,
        `reserved_room_type`, `agent`, `days_in_waiting_list`, `customer_type`,
        `adr`, `total_of_special_requests`, `country` atau `country_grouped`,
        `required_car_parking_spaces`, `previous_cancellations`,
        `previous_bookings_not_canceled`, `booking_changes`
        """)

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_batch = pd.read_csv(uploaded)
                st.success(f"File dimuat: **{len(df_batch):,} baris**")

                with st.spinner("Feature engineering & prediksi…"):
                    X_batch = _batch_feature_engineering(df_batch)
                    probs   = model.predict_proba(X_batch)[:, 1]
                    preds   = (probs >= THRESHOLD).astype(int)

                df_result = df_batch.copy()
                df_result["cancel_probability"] = probs.round(4)
                df_result["prediction"]  = preds
                df_result["risk_label"]  = pd.cut(
                    probs,
                    bins=[-0.01, 0.40, 0.65, 1.01],
                    labels=["🟢 Rendah", "🟡 Menengah", "🔴 Tinggi"]
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Booking",   f"{len(df_result):,}")
                c2.metric("Prediksi Cancel", f"{preds.sum():,}")
                c3.metric("% Cancel",        f"{preds.mean()*100:.1f}%")

                st.dataframe(
                    df_result[["cancel_probability", "prediction", "risk_label"]
                               + list(df_batch.columns)].head(200),
                    use_container_width=True
                )
                st.download_button(
                    "⬇️ Download Hasil (CSV)",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                )
            except Exception:
                st.error("Gagal memproses file CSV.")
                st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — SHAP GLOBAL
    # ══════════════════════════════════════════════════════════════
    with tab_global:
        st.subheader("📊 Feature Importance Global (Mean |SHAP|)")
        st.markdown(
            "10 fitur paling berpengaruh pada keputusan model — diambil dari "
            "LightGBM sebagai base estimator terkuat dalam Stacking."
        )
        st.plotly_chart(_shap_global_chart(), use_container_width=True, key="shap_global_tab")

        st.info(
            "ℹ️ Fitur bertanda `~` adalah estimasi berdasarkan urutan SHAP beeswarm "
            "(Section 9.5 notebook). Nilai pasti: `country_grouped`, "
            "`parking_bin_has_parking`, `lead_x_transient`. "
            "Update `GLOBAL_IMPORTANCE` di prediction.py setelah re-training."
        )
