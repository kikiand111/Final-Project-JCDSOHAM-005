"""Halaman Prediksi — single booking & batch CSV prediction."""

import math
import pickle
import traceback
import types
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "final_model_v5.pkl"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_TO_NUM: dict[str, int] = {m: i + 1 for i, m in enumerate(MONTHS)}

# Sinkron dengan training
SEASON_MAP: dict[int, str] = {
    12: "Winter", 1: "Winter",  2: "Winter",
     3: "Spring", 4: "Spring",  5: "Spring",
     6: "Summer", 7: "Summer",  8: "Summer",
     9: "Fall",  10: "Fall",   11: "Fall",
}

# Daftar negara sesuai country_grouped di training data
COUNTRIES = [
    "AGO", "ARG", "AUS", "AUT", "BEL", "BRA", "CHE", "CHN",
    "CZE", "DEU", "DNK", "DZA", "ESP", "FIN", "FRA", "GBR",
    "GRC", "HRV", "HUN", "IND", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LUX", "MAR", "NLD", "NOR", "Other", "POL", "PRT",
    "ROU", "RUS", "SWE", "TUR", "USA",
]

# Global SHAP importance — update sesuai hasil training
GLOBAL_IMPORTANCE: dict[str, float] = {
    "country_grouped":           0.412,
    "agent":                     0.298,
    "market_segment":            0.187,
    "lead_time":                 0.165,
    "adr":                       0.143,
    "total_of_special_requests": 0.138,
    "customer_type":             0.121,
    "distribution_channel":      0.099,
    "prev_cancel_bin":           0.087,
    "reserved_room_type":        0.074,
}

THRESHOLD: float = 0.35


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL — dua-pass unpickler untuk menangani Pipeline.dtype
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan: `{MODEL_PATH}`")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        return model

    except Exception:
        st.error("Gagal memuat model.")
        st.code(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────
# BUILD FEATURES  (29 kolom, urutan HARUS sama dengan pipeline)
# ─────────────────────────────────────────────────────────────────
def _month_sincos(month_num: int) -> tuple[float, float]:
    a = 2 * math.pi * month_num / 12
    return math.sin(a), math.cos(a)


def build_features(raw: dict) -> pd.DataFrame:
    month_num  = MONTH_TO_NUM[raw["arrival_date_month"]]
    month_sin, month_cos = _month_sincos(month_num)
    season     = SEASON_MAP[month_num]

    lead       = int(raw["lead_time"])
    ctype      = raw["customer_type"]
    lead_x_t   = lead * (1 if ctype == "Transient" else 0)

    stays_wk   = int(raw["stays_in_week_nights"])
    stays_we   = int(raw["stays_in_weekend_nights"])
    is_wknd    = int(stays_wk == 0 and stays_we > 0)

    country    = raw.get("country_grouped", "Other")
    if country not in COUNTRIES:
        country = "Other"

    no_commit  = int(raw.get("deposit_type", "Non Refund") == "No Deposit")

    row = {
        "hotel":                          raw["hotel"],
        "lead_time":                      lead,
        "stays_in_weekend_nights":        stays_we,
        "stays_in_week_nights":           stays_wk,
        "adults":                         int(raw["adults"]),
        "meal":                           raw["meal"],
        "market_segment":                 raw["market_segment"],
        "distribution_channel":           raw["distribution_channel"],
        "is_repeated_guest":              int(raw["is_repeated_guest"]),
        "reserved_room_type":             raw["reserved_room_type"],
        "agent":                          raw["agent"],
        "company":                        float("nan"),
        "days_in_waiting_list":           int(raw["days_in_waiting_list"]),
        "customer_type":                  ctype,
        "adr":                            float(raw["adr"]),
        "total_of_special_requests":      int(raw["total_of_special_requests"]),
        "country_grouped":                country,
        "arrival_season":                 season,
        "month_sin":                      month_sin,
        "month_cos":                      month_cos,
        "babies_bin":                     raw["babies_bin"],
        "children_bin":                   raw["children_bin"],
        "parking_bin":                    raw["parking_bin"],
        "prev_cancel_bin":                raw["prev_cancel_bin"],
        "prev_loyal_bin":                 raw["prev_loyal_bin"],
        "booking_changes_bin":            raw["booking_changes_bin"],
        "no_commitment":                  no_commit,
        "lead_x_transient":               lead_x_t,
        "is_weekend_only":                is_wknd,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────
# BATCH FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def _prev_cancel_bin(n):
    n = int(n) if pd.notna(n) else 0
    return "never" if n == 0 else ("once" if n == 1 else "multiple")

def _prev_loyal_bin(n):
    n = int(n) if pd.notna(n) else 0
    return "none" if n == 0 else ("some" if n <= 4 else "many")

def _booking_changes_bin(n):
    n = int(n) if pd.notna(n) else 0
    return "none" if n == 0 else ("some" if n <= 2 else "many")

def _batch_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month_nums = df["arrival_date_month"].map(MONTH_TO_NUM).fillna(1).astype(int)
    df["month_sin"]      = month_nums.apply(lambda m: math.sin(2*math.pi*m/12))
    df["month_cos"]      = month_nums.apply(lambda m: math.cos(2*math.pi*m/12))
    df["arrival_season"] = month_nums.map(SEASON_MAP)

    if "country" in df.columns and "country_grouped" not in df.columns:
        df["country_grouped"] = df["country"].apply(
            lambda c: c if c in COUNTRIES else "Other"
        )

    df["lead_x_transient"] = df["lead_time"].fillna(0) * (
        df["customer_type"].eq("Transient").astype(int)
    )
    df["is_weekend_only"] = (
        (df["stays_in_week_nights"].fillna(0) == 0) &
        (df["stays_in_weekend_nights"].fillna(0) > 0)
    ).astype(int)

    df["babies_bin"]          = df["babies"].fillna(0).apply(
        lambda x: "has_baby" if int(x) > 0 else "no_baby")
    df["children_bin"]        = df["children"].fillna(0).apply(
        lambda x: "has_children" if int(x) > 0 else "no_children")
    df["parking_bin"]         = df["required_car_parking_spaces"].fillna(0).apply(
        lambda x: "has_parking" if int(x) > 0 else "no_parking")
    df["prev_cancel_bin"]     = df["previous_cancellations"].fillna(0).apply(_prev_cancel_bin)
    df["prev_loyal_bin"]      = df["previous_bookings_not_canceled"].fillna(0).apply(_prev_loyal_bin)
    df["booking_changes_bin"] = df["booking_changes"].fillna(0).apply(_booking_changes_bin)

    if "deposit_type" in df.columns:
        df["no_commitment"] = (df["deposit_type"] == "No Deposit").astype(int)
    else:
        df["no_commitment"] = 0

    df["company"] = float("nan")
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
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]
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
    items = sorted(GLOBAL_IMPORTANCE.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[i[1] for i in items], y=[i[0] for i in items], orientation="h",
        marker_color="#7C5CBF",
        text=[f"{v:.3f}" for v in [i[1] for i in items]], textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Global — Fitur Terpenting di Seluruh Data Training",
        xaxis_title="Mean |SHAP Value|",
        height=400, margin=dict(t=50, b=20, l=210, r=80),
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
                country_grouped = st.selectbox("Negara Asal Tamu", COUNTRIES,
                                               index=COUNTRIES.index("PRT"))

            st.subheader("🤝 Agen & Deposit")
            c1, c2, c3 = st.columns(3)
            with c1:
                agent_input = st.number_input(
                    "ID Agen (0 = tanpa agen)",
                    min_value=0, max_value=535, value=0, step=1,
                    help="Kode numerik agen. 0 = tidak melalui agen. (0–535)"
                )
                agent_val = float(agent_input) if agent_input > 0 else float("nan")
            with c2:
                deposit_type = st.selectbox(
                    "Tipe Deposit",
                    ["No Deposit", "Non Refund", "Refundable"],
                    help="'No Deposit' → no_commitment = 1 (risiko lebih tinggi)"
                )
            with c3:
                parking_bin = st.selectbox(
                    "Parkir",
                    options=["no_parking", "has_parking"],
                    format_func=lambda x: "Tidak butuh parkir" if x == "no_parking" else "Butuh parkir",
                )

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
                        "never": "Tidak pernah (0)",
                        "once": "Satu kali (1)",
                        "multiple": "Lebih dari satu (≥2)"
                    }[x],
                )
            with c3:
                prev_loyal_bin = st.selectbox(
                    "Booking Sukses Sebelumnya",
                    options=["none", "some", "many"],
                    format_func=lambda x: {
                        "none": "Tidak ada (0)",
                        "some": "Sedikit (1–4)",
                        "many": "Banyak (≥5)"
                    }[x],
                )
            with c4:
                booking_changes_bin = st.selectbox(
                    "Perubahan Booking",
                    options=["none", "some", "many"],
                    format_func=lambda x: {
                        "none": "Tidak ada (0)",
                        "some": "Sedikit (1–2)",
                        "many": "Banyak (≥3)"
                    }[x],
                )

            submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

        if submitted:
            raw = {
                "hotel":               hotel,
                "lead_time":           lead_time,
                "stays_in_weekend_nights": stays_weekend,
                "stays_in_week_nights":    stays_week,
                "adults":              adults,
                "meal":                meal,
                "market_segment":      market_segment,
                "distribution_channel":distribution_channel,
                "is_repeated_guest":   is_repeated_val,
                "reserved_room_type":  reserved_room,
                "agent":               agent_val,
                "days_in_waiting_list":days_waiting,
                "customer_type":       customer_type,
                "adr":                 adr,
                "total_of_special_requests": total_special,
                "country_grouped":     country_grouped,
                "arrival_date_month":  arrival_month,
                "deposit_type":        deposit_type,
                "babies_bin":          babies_bin,
                "children_bin":        children_bin,
                "parking_bin":         parking_bin,
                "prev_cancel_bin":     prev_cancel_bin,
                "prev_loyal_bin":      prev_loyal_bin,
                "booking_changes_bin": booking_changes_bin,
            }

                            # =========================
            # PREDIKSI
            # =========================
            try:
                X = build_features(raw)
                prob = float(model.predict_proba(X)[0, 1])
                pred = int(prob >= THRESHOLD)

                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")

                col_g, col_r = st.columns([1.2, 1])

                with col_g:
                    st.plotly_chart(_gauge_chart(prob), use_container_width=True)

                with col_r:
                    st.markdown(f"**Probabilitas Pembatalan:** `{prob:.1%}`")
                    st.markdown(f"**Threshold:** `{THRESHOLD:.0%}`")

                    label = "🔴 **AKAN DIBATALKAN**" if pred == 1 else "🟢 **TIDAK DIBATALKAN**"
                    st.markdown(f"**Prediksi:** {label}")

                    season_disp = SEASON_MAP[MONTH_TO_NUM[arrival_month]]
                    nc = "⚠️ Ya" if deposit_type == "No Deposit" else "Tidak"
                    we = "Ya" if stays_week == 0 and stays_weekend > 0 else "Tidak"

                    st.caption(
                        f"Musim: **{season_disp}** · No-commitment: **{nc}** · "
                        f"Weekend only: **{we}**"
                    )

                st.markdown("---")
                _recommendation(prob, pred)

            except Exception:
                st.error("Terjadi kesalahan saat prediksi.")
                st.code(traceback.format_exc())


            # =========================
            # SHAP (WAJIB DI LUAR TRY)
            # =========================
            with st.expander("🔍 SHAP Lokal — Detail Pengaruh Fitur", expanded=False):
                try:
                    import lightgbm as lgb
                    import numpy as np

                    # =========================================================
                    # 1. Cari LightGBM dari stacking
                    # =========================================================
                    stacking = None
                    for _, step in model.steps:
                        if hasattr(step, "estimators_"):
                            stacking = step
                            break

                    lgbm_pipe = None
                    if stacking is not None:
                        for est in stacking.estimators_:
                            if hasattr(est, "named_steps"):
                                for _, sobj in est.named_steps.items():
                                    if isinstance(sobj, lgb.LGBMClassifier):
                                        lgbm_pipe = est
                                        break
                            elif isinstance(est, lgb.LGBMClassifier):
                                lgbm_pipe = est

                            if lgbm_pipe is not None:
                                break

                    if lgbm_pipe is None:
                        st.warning("LightGBM tidak ditemukan.")
                    else:
                        # =========================================================
                        # 2. Ambil classifier
                        # =========================================================
                        if hasattr(lgbm_pipe, "named_steps"):
                            clf_lgbm = None
                            for _, sobj in lgbm_pipe.named_steps.items():
                                if isinstance(sobj, lgb.LGBMClassifier):
                                    clf_lgbm = sobj
                                    break
                        else:
                            clf_lgbm = lgbm_pipe

                        if clf_lgbm is None:
                            st.warning("Classifier tidak ditemukan.")
                        else:
                            # =========================================================
                            # 3. PREPROCESSING (AMBIL FEATURE NAME)
                            # =========================================================
                            ct_shap = model.named_steps["preprocessor"]
                            X_t = ct_shap.transform(X)

                            def get_ct_feature_names(ct):
                                output_features = []

                                for name, trans, cols in ct.transformers_:
                                    if name == "remainder":
                                        continue

                                    if hasattr(trans, "get_feature_names_out"):
                                        try:
                                            names = trans.get_feature_names_out(cols)
                                        except:
                                            names = cols
                                    else:
                                        names = cols

                                    output_features.extend(names)

                                return output_features

                            # ✅ INI YANG KAMU KURANGIN TADI
                            feat_names = get_ct_feature_names(ct_shap)

                            # optional: bikin lebih readable
                            feat_names = [
                                f.replace("country_grouped_", "Country: ")
                                .replace("deposit_type_", "Deposit: ")
                                .replace("market_segment_", "Segment: ")
                                .replace("distribution_channel_", "Channel: ")
                                for f in feat_names
                            ]

                            if hasattr(X_t, "toarray"):
                                X_t = X_t.toarray()

                            # =========================================================
                            # 4. SHAP
                            # =========================================================
                            explainer = shap.TreeExplainer(clf_lgbm)
                            sv = explainer.shap_values(X_t)

                            if isinstance(sv, list):
                                sv = sv[1]

                            sv = sv[0] if sv.ndim == 2 else sv

                            # =========================================================
                            # 5. VISUAL
                            # =========================================================
                            st.plotly_chart(
                                _shap_local_chart(sv, feat_names),
                                use_container_width=True
                            )

                            # =========================================================
                            # 6. INTERPRETASI OTOMATIS (INI YANG KAMU MAU)
                            # =========================================================
                            st.markdown("### 📊 Penjelasan Model")

                            top_idx = np.argsort(np.abs(sv))[::-1][:5]

                            for i in top_idx:
                                direction = "meningkatkan" if sv[i] > 0 else "menurunkan"

                                st.write(
                                    f"• **{feat_names[i]}** → {direction} kemungkinan cancel "
                                    f"({sv[i]:.3f})"
                                )

                except Exception:
                    st.warning("SHAP gagal.")
                    st.code(traceback.format_exc())
    # ══════════════════════════════════════════════════════════════
    # TAB 2 — BATCH
    # ══════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("""
        Upload CSV dengan kolom:
        `hotel`, `lead_time`, `arrival_date_month`, `stays_in_weekend_nights`,
        `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`,
        `market_segment`, `distribution_channel`, `is_repeated_guest`,
        `reserved_room_type`, `agent`, `days_in_waiting_list`, `customer_type`,
        `adr`, `total_of_special_requests`, `country` atau `country_grouped`,
        `required_car_parking_spaces`, `previous_cancellations`,
        `previous_bookings_not_canceled`, `booking_changes`, `deposit_type`
        """)

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_batch = pd.read_csv(uploaded)
                st.success(f"File dimuat: **{len(df_batch):,} baris**")

                with st.spinner("Feature engineering & prediksi…"):
                    X_batch = _batch_feature_engineering(df_batch)
                    probs = model.predict_proba(X_batch)[:, 1]
                    preds = (probs >= THRESHOLD).astype(int)

                df_result = df_batch.copy()
                df_result["cancel_probability"] = probs.round(4)
                df_result["prediction"] = preds
                df_result["risk_label"] = pd.cut(
                    probs, bins=[-0.01, 0.40, 0.65, 1.01],
                    labels=["🟢 Rendah", "🟡 Menengah", "🔴 Tinggi"]
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Booking",   f"{len(df_result):,}")
                c2.metric("Prediksi Cancel", f"{preds.sum():,}")
                c3.metric("% Cancel",        f"{preds.mean()*100:.1f}%")

                st.dataframe(
                    df_result[["cancel_probability","prediction","risk_label"]
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
            "10 fitur paling berpengaruh pada keputusan model secara keseluruhan."
        )
        st.plotly_chart(_shap_global_chart(), use_container_width=True)
        st.caption("Update `GLOBAL_IMPORTANCE` di baris atas file ini jika model di-retrain.")
