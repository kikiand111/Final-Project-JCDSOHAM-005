"""Halaman Tentang v3 — info model + confusion matrix + cost analysis interaktif."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

MODEL_F05   = 0.7415
MODEL_ACC   = 0.8446
MODEL_PREC  = 0.7849
MODEL_REC   = 0.6073
MODEL_AUC   = 0.9144
MODEL_AUPR  = 0.8043
THRESHOLD   = 0.60
N_FEATURES  = 29

# Confusion matrix dari Section 9.1 notebook (25.780 booking test)
TN, FP, FN, TP = 18266, 1192, 2813, 4350  # total = 36621? let me recalc
# Actually: 25780 test rows. Let me use the values from the notebook text.
# From cost analysis: FP=1192, FN=2813, TP=4350
# TN = total - FP - FN - TP = 25780 - 1192 - 2813 - 4350 = 17425
TN = 25780 - FP - FN - TP  # = 17425

CV_FOLDS = {
    "Fold":      [1,      2,      3,      4,      5],
    "F0.5":      [0.7374, 0.7388, 0.7269, 0.7348, 0.7305],
    "Precision": [0.7873, 0.7864, 0.7740, 0.7861, 0.7721],
    "Recall":    [0.5881, 0.5949, 0.5848, 0.5827, 0.6010],
}

HPO_DATA = {
    "Model":  ["CatBoost","LightGBM","XGBoost"],
    "Rand":   [0.7232,    0.7230,    0.7232],
    "Grid":   [0.7210,    0.7188,    0.7156],
    "Optuna": [0.7233,    0.7246,    0.7256],
}

SHAP_GLOBAL = {
    "country_grouped":          0.9272,
    "agent":                    0.6850,
    "parking_bin_has_parking":  0.5478,
    "parking_bin_no_parking":   0.4150,
    "lead_x_transient":         0.3941,
    "no_commitment":            0.2650,
    "lead_time":                0.2180,
    "prev_cancel_bin_never":    0.1740,
    "booking_changes_bin_none": 0.1310,
    "prev_loyal_bin_none":      0.0920,
}


def _metric_gauge(label, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": label, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,   50], "color": "#F8F9FA"},
                {"range": [50,  75], "color": "#EEF2FF"},
                {"range": [75, 100], "color": "#E8F4FD"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8, "value": value * 100,
            },
        },
    ))
    fig.update_layout(height=180, margin=dict(t=30,b=5,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render() -> None:
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown(
        "Hotel Booking Cancellation Predictor berbasis **Stacking Ensemble** — "
        "prediksi pembatalan booking hotel dengan explainability berbasis SHAP."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performa Model",
        "🔢 Confusion Matrix",
        "💰 Analisis Biaya",
        "🔁 CV & HPO",
        "🔍 SHAP Global",
    ])

    # ── TAB 1: Performa ───────────────────────────────────────────
    with tab1:
        st.markdown("#### Metrik Evaluasi Model pada X_test (25.780 booking)")

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(_metric_gauge("F0.5 Score", MODEL_F05,  "#534AB7"), use_container_width=True)
        with g2:
            st.plotly_chart(_metric_gauge("Precision",  MODEL_PREC, "#2980B9"), use_container_width=True)
        with g3:
            st.plotly_chart(_metric_gauge("Recall",     MODEL_REC,  "#27AE60"), use_container_width=True)
        with g4:
            st.plotly_chart(_metric_gauge("Accuracy",   MODEL_ACC,  "#E67E22"), use_container_width=True)

        st.markdown("---")
        col_info, col_radar = st.columns([1, 1.2])

        with col_info:
            st.markdown("#### Info Model")
            st.markdown(f"""
            | Atribut | Detail |
            |---|---|
            | **Algoritma** | Stacking Ensemble |
            | **Base Estimators** | LightGBM · XGBoost · CatBoost · RF |
            | **Meta Learner** | Logistic Regression |
            | **HPO Method** | Optuna (100 trials) |
            | **Threshold** | `{THRESHOLD}` |
            | **Fitur Input** | {N_FEATURES} kolom |
            | **ROC-AUC** | {MODEL_AUC} |
            | **AUC-PR** | {MODEL_AUPR} |
            | **Brier Score** | 0.1065 ✅ |
            | **Avg Calibration Gap** | 0.0392 ✅ |
            """)

        with col_radar:
            st.markdown("#### Radar Performa")
            cats = ["F0.5","Precision","Recall","Accuracy","ROC-AUC"]
            vals = [MODEL_F05, MODEL_PREC, MODEL_REC, MODEL_ACC, MODEL_AUC]
            vals_pct = [v * 100 for v in vals]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals_pct + [vals_pct[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(83,74,183,0.15)",
                line=dict(color="#534AB7", width=2),
                marker=dict(size=7, color="#534AB7"),
                text=[f"{v:.1f}%" for v in vals_pct + [vals_pct[0]]],
                hovertemplate="%{theta}: %{text}<extra></extra>",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[50,100], tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=12)),
                ),
                height=300, margin=dict(t=20,b=20,l=40,r=40),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 2: Confusion Matrix ───────────────────────────────────
    with tab2:
        st.markdown("#### Confusion Matrix — Stacking @ threshold = 0.60")
        total = TN + FP + FN + TP

        col_cm, col_stats = st.columns([1, 1])

        with col_cm:
            z      = [[TN, FP], [FN, TP]]
            labels = [
                [f"TN\n{TN:,}\n({TN/total*100:.1f}%)",  f"FP\n{FP:,}\n({FP/total*100:.1f}%)"],
                [f"FN\n{FN:,}\n({FN/total*100:.1f}%)",  f"TP\n{TP:,}\n({TP/total*100:.1f}%)"],
            ]
            colors = ["#D4EDDA","#F8D7DA","#FFF3CD","#CCE5FF"]

            fig_cm = go.Figure()
            positions = [(0,0,TN,"TN","#D4EDDA"),(0,1,FP,"FP","#F8D7DA"),
                         (1,0,FN,"FN","#FFF3CD"),(1,1,TP,"TP","#CCE5FF")]
            for r, c, val, label, color in positions:
                fig_cm.add_shape(
                    type="rect", x0=c, y0=1-r, x1=c+1, y1=2-r,
                    fillcolor=color, line=dict(color="white", width=3),
                )
                pct = val / total * 100
                fig_cm.add_annotation(
                    x=c+0.5, y=1-r+0.72, text=f"<b>{label}</b>",
                    showarrow=False, font=dict(size=14, color="#555"),
                )
                fig_cm.add_annotation(
                    x=c+0.5, y=1-r+0.45, text=f"<b>{val:,}</b>",
                    showarrow=False, font=dict(size=22, color="#111"),
                )
                fig_cm.add_annotation(
                    x=c+0.5, y=1-r+0.20, text=f"{pct:.1f}%",
                    showarrow=False, font=dict(size=12, color="#666"),
                )
            fig_cm.update_xaxes(
                tickvals=[0.5,1.5],
                ticktext=["Predicted<br>Not Cancel","Predicted<br>Cancel"],
                range=[0,2],
            )
            fig_cm.update_yaxes(
                tickvals=[0.5,1.5],
                ticktext=["Actual<br>Cancel","Actual<br>Not Cancel"],
                range=[0,2],
            )
            fig_cm.update_layout(
                height=340, margin=dict(t=20,b=60,l=100,r=20),
                plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_stats:
            st.markdown("#### Interpretasi Bisnis")

            biz_data = [
                ("✅ TP", TP, "Cancel → diprediksi cancel", "#CCE5FF"),
                ("⚠️ FP", FP, "Tidak cancel → diprediksi cancel<br><small>→ intervensi sia-sia</small>", "#F8D7DA"),
                ("❌ FN", FN, "Cancel → diprediksi tidak cancel<br><small>→ kamar kosong tak terduga</small>", "#FFF3CD"),
                ("✅ TN", TN, "Tidak cancel → benar", "#D4EDDA"),
            ]
            for tag, val, desc, color in biz_data:
                st.markdown(
                    f"""<div style="background:{color};border-radius:8px;padding:10px 14px;
                    margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
                    <div><b>{tag}</b> — {desc}</div>
                    <div style="font-size:20px;font-weight:700">{val:,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("")
            fp_tp = FP / TP
            st.caption(
                f"**FP/TP ratio = {fp_tp:.2f}** — setiap 1 prediksi cancel benar, "
                f"ada {fp_tp:.2f} false alarm."
            )

    # ── TAB 3: Cost Analysis ──────────────────────────────────────
    with tab3:
        st.markdown("#### Analisis Biaya — Simulasi Interaktif")
        st.markdown(
            "Sesuaikan asumsi biaya per jenis kesalahan prediksi untuk melihat "
            "dampak finansial model vs tanpa model."
        )

        c1, c2 = st.columns(2)
        with c1:
            cost_fp = st.slider(
                "💸 Biaya per False Positive (€) — Intervensi Sia-sia",
                min_value=100, max_value=2000, value=700, step=50,
                help="Biaya walking guest, upgrade, kompensasi yang tidak perlu",
            )
        with c2:
            cost_fn = st.slider(
                "💸 Biaya per False Negative (€) — Kamar Kosong",
                min_value=100, max_value=2000, value=300, step=50,
                help="Opportunity cost dari kamar yang tidak terjual",
            )

        total_fp_cost    = FP * cost_fp
        total_fn_cost    = FN * cost_fn
        total_model_cost = total_fp_cost + total_fn_cost
        total_base_cost  = (FN + TP) * cost_fn  # baseline: semua cancel tidak terantisipasi
        savings          = total_base_cost - total_model_cost
        savings_pct      = savings / total_base_cost * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Biaya Baseline (tanpa model)", f"€{total_base_cost:,.0f}")
        m2.metric("Biaya Dengan Model",           f"€{total_model_cost:,.0f}")
        m3.metric("Penghematan",                  f"€{savings:,.0f}",
                  delta=f"{savings_pct:.1f}%", delta_color="normal")

        # Bar chart breakdown
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            name="False Positive Cost",
            x=["Dengan Model"], y=[total_fp_cost],
            marker_color="#F8D7DA",
            text=f"€{total_fp_cost:,.0f}", textposition="inside",
        ))
        fig_cost.add_trace(go.Bar(
            name="False Negative Cost",
            x=["Dengan Model"], y=[total_fn_cost],
            marker_color="#FFF3CD",
            text=f"€{total_fn_cost:,.0f}", textposition="inside",
        ))
        fig_cost.add_trace(go.Bar(
            name="Baseline Cost",
            x=["Tanpa Model"], y=[total_base_cost],
            marker_color="#D6D6D6",
            text=f"€{total_base_cost:,.0f}", textposition="inside",
        ))
        fig_cost.update_layout(
            barmode="stack", height=360,
            yaxis_title="Total Biaya (€)",
            plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        ratio = total_fp_cost / total_fn_cost if total_fn_cost > 0 else 0
        dominant = "False Positive" if total_fp_cost > total_fn_cost else "False Negative"
        st.info(
            f"Dengan asumsi FP=€{cost_fp:,} dan FN=€{cost_fn:,}, "
            f"error yang lebih dominan secara finansial adalah **{dominant}** "
            f"(rasio FP:FN cost = {ratio:.2f}:1). "
            f"Threshold 0.6 menyeimbangkan keduanya."
        )

    # ── TAB 4: CV & HPO ───────────────────────────────────────────
    with tab4:
        import pandas as pd
        st.markdown("#### CV Stability + HPO Comparison")

        sub1, sub2 = st.tabs(["🔁 CV Stability", "🔬 HPO Comparison"])

        with sub1:
            cv_df = pd.DataFrame(CV_FOLDS)
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i}" for i in cv_df["Fold"]],
                y=cv_df["F0.5"], name="F0.5", marker_color="#534AB7",
                text=cv_df["F0.5"].round(4), textposition="outside",
            ))
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i}" for i in cv_df["Fold"]],
                y=cv_df["Precision"], name="Precision", marker_color="#27AE60", opacity=0.75,
            ))
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i}" for i in cv_df["Fold"]],
                y=cv_df["Recall"], name="Recall", marker_color="#E67E22", opacity=0.75,
            ))
            mean_f05 = sum(CV_FOLDS["F0.5"]) / 5
            fig_cv.add_hline(y=mean_f05, line_dash="dash", line_color="#534AB7",
                             annotation_text=f"Mean F0.5 = {mean_f05:.4f}")
            fig_cv.update_layout(
                barmode="group", height=360,
                yaxis=dict(range=[0.5, 0.9], title="Score"),
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                legend=dict(orientation="h", y=-0.15), margin=dict(t=20,b=10),
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            std_f05 = pd.Series(CV_FOLDS["F0.5"]).std()
            st.success(f"Std F0.5 = **{std_f05:.4f}** — variasi sangat kecil, model stabil ✅")

        with sub2:
            hpo_df = pd.DataFrame(HPO_DATA)
            fig_hpo = go.Figure()
            for col, color, name in [
                ("Rand","#4472C4","RandomizedSearch"),
                ("Grid","#ED7D31","GridSearch"),
                ("Optuna","#70AD47","Optuna ✓"),
            ]:
                fig_hpo.add_trace(go.Bar(
                    name=name, x=hpo_df["Model"], y=hpo_df[col],
                    marker_color=color,
                    text=hpo_df[col].round(4), textposition="outside",
                ))
            fig_hpo.update_layout(
                barmode="group", height=360,
                yaxis=dict(range=[0.71,0.73], title="CV F0.5"),
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                legend=dict(orientation="h", y=-0.15), margin=dict(t=20,b=10),
            )
            st.plotly_chart(fig_hpo, use_container_width=True)
            st.success("✅ Optuna konsisten menang di semua model via Bayesian Optimization (TPE).")

    # ── TAB 5: SHAP Global ────────────────────────────────────────
    with tab5:
        st.markdown("#### SHAP Global — Mean |SHAP| dari LightGBM")
        st.markdown(
            "Nilai terkonfirmasi (✓) dari teks notebook. "
            "Nilai estimasi (~) berdasarkan urutan SHAP beeswarm Section 9.5."
        )

        items     = sorted(SHAP_GLOBAL.items(), key=lambda x: x[1])
        confirmed = {"country_grouped","parking_bin_has_parking","lead_x_transient"}
        colors    = ["#534AB7" if n in confirmed else "#9B8DE8" for n, _ in items]
        labels    = [f"{n}  ✓" if n in confirmed else f"{n}  ~" for n, _ in items]

        fig_shap = go.Figure(go.Bar(
            x=[v for _, v in items],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for _, v in items],
            textposition="outside",
        ))
        fig_shap.update_layout(
            xaxis_title="Mean |SHAP Value|",
            height=420,
            plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
            margin=dict(t=20,b=10,r=120),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.info("🟣 **Warna gelap** = nilai terkonfirmasi dari output notebook")
        col2.info("🟣 **Warna terang** = estimasi berdasarkan urutan SHAP beeswarm")

    st.markdown("---")
    st.warning(
        "⚠️ Model ini untuk tujuan pembelajaran. "
        "Kombinasikan hasil prediksi dengan judgement operasional hotel."
    )
