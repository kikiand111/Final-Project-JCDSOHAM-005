"""Halaman Beranda v3 — statistik interaktif dengan filter hotel."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_F05  = "0.7415"
MODEL_PREC = "78.49%"
MODEL_REC  = "60.73%"
THRESHOLD  = 0.60

MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("hotel_bookings.csv")


@st.cache_data
def compute_all(df: pd.DataFrame) -> dict:
    monthly = (
        df.groupby(["arrival_date_month","hotel"])["is_canceled"]
        .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
    )
    monthly["cr"] *= 100
    monthly["arrival_date_month"] = pd.Categorical(
        monthly["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    monthly = monthly.sort_values("arrival_date_month")

    hotel_stats = (
        df.groupby("hotel")["is_canceled"]
        .agg(["mean","count","sum"])
        .rename(columns={"mean":"cr","count":"total","sum":"canceled"})
        .reset_index()
    )
    hotel_stats["cr"] *= 100
    hotel_stats["not_canceled"] = hotel_stats["total"] - hotel_stats["canceled"]

    country_cr = (
        df.groupby(["country","hotel"])["is_canceled"]
        .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
    )
    country_cr["cr"] *= 100

    segment_cr = (
        df.groupby(["market_segment","hotel"])["is_canceled"]
        .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
    )
    segment_cr["cr"] *= 100

    lead_bins = pd.cut(
        df["lead_time"],
        bins=[-1,7,30,90,180,709],
        labels=["0–7","8–30","31–90","91–180","181+"]
    )
    lead_cr = (
        df.assign(lead_bin=lead_bins)
        .groupby(["lead_bin","hotel"])["is_canceled"]
        .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
    )
    lead_cr["cr"] *= 100

    return {
        "total":       len(df),
        "cancel_rate": df["is_canceled"].mean() * 100,
        "hotel_stats": hotel_stats,
        "monthly":     monthly,
        "country_cr":  country_cr,
        "segment_cr":  segment_cr,
        "lead_cr":     lead_cr,
    }


def render() -> None:
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown(
        "Tool prediksi pembatalan booking hotel berbasis **Stacking Ensemble** "
        "(LightGBM + XGBoost + CatBoost). Mendukung prediksi tunggal maupun batch CSV."
    )
    st.markdown("---")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("File `hotel_bookings.csv` tidak ditemukan.")
        return

    s = compute_all(df)

    # ── METRIC CARDS ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Booking",          f"{s['total']:,}")
    c2.metric("Cancel Rate Historis",   f"{s['cancel_rate']:.1f}%")
    c3.metric("F0.5 Score Model",       MODEL_F05)
    c4.metric("Precision",              MODEL_PREC)
    c5.metric("Threshold Keputusan",    f"{THRESHOLD:.0%}")

    st.markdown("---")

    # ── FILTER GLOBAL ─────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        hotel_filter = st.selectbox(
            "🏨 Filter Hotel",
            options=["Semua Hotel", "City Hotel", "Resort Hotel"],
            help="Filter berlaku untuk semua tab di bawah",
        )

    if hotel_filter == "Semua Hotel":
        df_f = df.copy()
        global_cr = s["cancel_rate"]
    else:
        df_f = df[df["hotel"] == hotel_filter].copy()
        global_cr = df_f["is_canceled"].mean() * 100

    with col_f2:
        st.markdown(
            f"<div style='padding-top:32px; color:#888; font-size:13px'>"
            f"Menampilkan <b>{len(df_f):,}</b> booking · "
            f"Cancel rate: <b>{global_cr:.1f}%</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Tren Bulanan",
        "🏨 Per Hotel",
        "🌍 Negara",
        "📢 Segmen Pasar",
        "⏱️ Lead Time",
    ])

    # ── TAB 1: Tren Bulanan ───────────────────────────────────────
    with tab1:
        st.markdown(f"#### Cancel Rate per Bulan — *{hotel_filter}*")

        monthly_f = (
            df_f.groupby("arrival_date_month")["is_canceled"]
            .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
        )
        monthly_f["cr"] *= 100
        monthly_f["arrival_date_month"] = pd.Categorical(
            monthly_f["arrival_date_month"], categories=MONTH_ORDER, ordered=True
        )
        monthly_f = monthly_f.sort_values("arrival_date_month")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_f["arrival_date_month"],
            y=monthly_f["cr"],
            marker_color=[
                "#E74C3C" if v >= 35 else "#F39C12" if v >= 28 else "#27AE60"
                for v in monthly_f["cr"]
            ],
            text=monthly_f["cr"].round(1).astype(str) + "%",
            textposition="outside",
            customdata=monthly_f["n"],
            hovertemplate="<b>%{x}</b><br>Cancel Rate: %{y:.1f}%<br>Jumlah Booking: %{customdata:,}<extra></extra>",
        ))
        fig.add_hline(
            y=global_cr, line_dash="dash", line_color="#534AB7",
            annotation_text=f"Rata-rata {global_cr:.1f}%",
            annotation_position="top right",
        )
        fig.update_layout(
            yaxis_title="Cancel Rate (%)", height=380,
            plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
            margin=dict(t=30, b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 ≥35% &nbsp;&nbsp; 🟡 28–35% &nbsp;&nbsp; 🟢 <28%")

    # ── TAB 2: Per Hotel ──────────────────────────────────────────
    with tab2:
        st.markdown("#### Perbandingan City Hotel vs Resort Hotel")
        col_a, col_b = st.columns(2)

        hs = s["hotel_stats"]
        with col_a:
            fig_bar = go.Figure(go.Bar(
                x=hs["hotel"], y=hs["cr"],
                marker_color=["#534AB7","#9B59B6"],
                text=hs["cr"].round(1).astype(str) + "%",
                textposition="outside",
                customdata=hs["total"],
                hovertemplate="<b>%{x}</b><br>Cancel Rate: %{y:.1f}%<br>Total Booking: %{customdata:,}<extra></extra>",
            ))
            fig_bar.add_hline(y=s["cancel_rate"], line_dash="dash", line_color="gray",
                              annotation_text=f"Avg {s['cancel_rate']:.1f}%")
            fig_bar.update_layout(
                title="Cancel Rate per Hotel",
                yaxis_title="Cancel Rate (%)", height=320,
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                margin=dict(t=40,b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(
                name="Tidak Batal", x=hs["hotel"], y=hs["not_canceled"],
                marker_color="#27AE60",
            ))
            fig_stack.add_trace(go.Bar(
                name="Batal", x=hs["hotel"], y=hs["canceled"],
                marker_color="#E74C3C",
            ))
            fig_stack.update_layout(
                barmode="stack", title="Komposisi Booking per Hotel",
                yaxis_title="Jumlah Booking", height=320,
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                margin=dict(t=40,b=10), legend=dict(orientation="h",y=-0.2),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        mc1, mc2 = st.columns(2)
        for col, (_, row) in zip([mc1, mc2], hs.iterrows()):
            col.metric(
                label=row["hotel"],
                value=f"{row['cr']:.1f}% cancel rate",
                delta=f"{row['cr'] - s['cancel_rate']:+.1f}pp vs rata-rata",
                delta_color="inverse",
            )

    # ── TAB 3: Negara ─────────────────────────────────────────────
    with tab3:
        st.markdown(f"#### Top 10 Negara Cancel Rate Tertinggi — *{hotel_filter}* *(min. 200 booking)*")

        country_f = (
            df_f.groupby("country")["is_canceled"]
            .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
        )
        country_f["cr"] *= 100
        country_f = country_f[country_f["n"] >= 200].nlargest(10,"cr")

        if country_f.empty:
            st.info("Tidak ada negara dengan ≥200 booking pada filter ini.")
        else:
            fig = px.bar(
                country_f.sort_values("cr"),
                x="cr", y="country", orientation="h",
                color="cr",
                color_continuous_scale=["#27AE60","#F39C12","#E74C3C"],
                text=country_f.sort_values("cr")["cr"].round(1).astype(str) + "%",
                labels={"cr":"Cancel Rate (%)","country":"Negara","n":"Jumlah Booking"},
                hover_data=["n"],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=380, coloraxis_showscale=False,
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                margin=dict(t=20,b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("ℹ️ Negara asal = fitur SHAP Global tertinggi (0.9272) dalam model.")

    # ── TAB 4: Segmen Pasar ───────────────────────────────────────
    with tab4:
        st.markdown(f"#### Cancel Rate per Segmen Pasar — *{hotel_filter}*")

        seg_f = (
            df_f.groupby("market_segment")["is_canceled"]
            .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
        )
        seg_f["cr"] *= 100
        seg_f = seg_f.sort_values("cr", ascending=False)

        fig = px.bar(
            seg_f, x="market_segment", y="cr",
            color="cr",
            color_continuous_scale=["#27AE60","#F39C12","#E74C3C"],
            text=seg_f["cr"].round(1).astype(str) + "%",
            labels={"cr":"Cancel Rate (%)","market_segment":"Segmen","n":"Jumlah Booking"},
            hover_data=["n"],
        )
        fig.update_traces(textposition="outside")
        fig.add_hline(y=global_cr, line_dash="dash", line_color="#534AB7",
                      annotation_text=f"Rata-rata {global_cr:.1f}%")
        fig.update_layout(
            height=360, coloraxis_showscale=False,
            plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
            margin=dict(t=30,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: Lead Time ──────────────────────────────────────────
    with tab5:
        st.markdown(f"#### Cancel Rate per Bucket Lead Time — *{hotel_filter}*")
        st.markdown(
            "Lead time × tipe tamu Transient (`lead_x_transient`) adalah fitur SHAP #5 "
            "dengan mean |SHAP| = 0.3941. "
            "Tab ini menunjukkan pola monotonic cancel rate seiring lead time."
        )

        lead_bins = pd.cut(
            df_f["lead_time"],
            bins=[-1,7,30,90,180,709],
            labels=["0–7 hari","8–30 hari","31–90 hari","91–180 hari","181+ hari"]
        )
        lead_f = (
            df_f.assign(lead_bin=lead_bins)
            .groupby("lead_bin", observed=True)["is_canceled"]
            .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
        )
        lead_f["cr"] *= 100

        col_lead, col_ctype = st.columns(2)

        with col_lead:
            fig_lead = go.Figure(go.Bar(
                x=lead_f["lead_bin"].astype(str),
                y=lead_f["cr"],
                marker_color=[
                    "#E74C3C" if v >= 35 else "#F39C12" if v >= 25 else "#27AE60"
                    for v in lead_f["cr"]
                ],
                text=lead_f["cr"].round(1).astype(str) + "%",
                textposition="outside",
                customdata=lead_f["n"],
                hovertemplate="<b>%{x}</b><br>Cancel Rate: %{y:.1f}%<br>N: %{customdata:,}<extra></extra>",
            ))
            fig_lead.add_hline(y=global_cr, line_dash="dash", line_color="#534AB7",
                               annotation_text=f"Avg {global_cr:.1f}%")
            fig_lead.update_layout(
                title="Cancel Rate per Lead Time Bucket",
                yaxis_title="Cancel Rate (%)", height=340,
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                margin=dict(t=40,b=10),
            )
            st.plotly_chart(fig_lead, use_container_width=True)

        with col_ctype:
            # Cancel rate Transient vs non-Transient per lead bucket
            lead_ctype = (
                df_f.assign(
                    lead_bin=lead_bins,
                    is_transient=(df_f["customer_type"] == "Transient")
                )
                .groupby(["lead_bin","is_transient"], observed=True)["is_canceled"]
                .agg(["mean","count"]).rename(columns={"mean":"cr","count":"n"}).reset_index()
            )
            lead_ctype["cr"] *= 100
            lead_ctype["Tipe"] = lead_ctype["is_transient"].map(
                {True:"Transient", False:"Non-Transient"}
            )

            fig_ct = px.line(
                lead_ctype, x="lead_bin", y="cr", color="Tipe",
                markers=True,
                color_discrete_map={"Transient":"#E74C3C","Non-Transient":"#2980B9"},
                labels={"cr":"Cancel Rate (%)","lead_bin":"Lead Time","n":"Jumlah"},
                hover_data=["n"],
            )
            fig_ct.update_layout(
                title="Transient vs Non-Transient per Lead Time",
                yaxis_title="Cancel Rate (%)", height=340,
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                margin=dict(t=40,b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ct, use_container_width=True)

        st.caption(
            "Tamu **Transient** dengan lead time panjang adalah kombinasi risiko tertinggi. "
            "Non-Transient (Contract/Group) tidak menunjukkan pola serupa karena terikat kontrak."
        )

    st.markdown("---")
    st.info("💡 **Mulai prediksi:** klik menu **🔮 Prediksi** di sidebar kiri.")
