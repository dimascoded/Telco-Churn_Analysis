"""
Telco Customer Churn — Executive & Retention Dashboard
Dibangun dari pipeline notebook (assessing → cleaning → EDA) dan data `telco_churn_cleaned.csv`.

Jalankan:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI & TEMA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "telco_churn_cleaned.csv"

C_CHURN = "#E5484D"    # merah  → churn / risiko
C_STAY = "#2A9D8F"     # teal   → bertahan
C_NEUTRAL = "#B8C0D0"  # abu    → di bawah rata-rata
C_NAVY = "#1F2A44"
C_MUTED = "#6B7690"

TENURE_ORDER = ["0–12 bln", "13–24 bln", "25–48 bln", "49–72 bln"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CHARGE_ORDER = ["< $35", "$35–55", "$55–75", "$75–95", "≥ $95"]
ADDONS = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
CAT_COLS = [
    "gender", "Senior", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }}
      /* Header */
      .hero {{ background: linear-gradient(120deg, {C_NAVY} 0%, #2F4B8F 100%);
               padding: 22px 28px; border-radius: 16px; color: #fff; margin-bottom: 18px; }}
      .hero h1 {{ margin: 0; font-size: 1.65rem; font-weight: 700; color: #fff; }}
      .hero p  {{ margin: 4px 0 0; opacity: .85; font-size: .92rem; }}
      /* KPI cards */
      [data-testid="stMetric"] {{ background: #fff; border: 1px solid #E6E9F2; border-radius: 14px;
                                  padding: 14px 18px; box-shadow: 0 1px 3px rgba(31,42,68,.05); }}
      [data-testid="stMetricLabel"] p {{ color: {C_MUTED}; font-size: .8rem; font-weight: 600;
                                         text-transform: uppercase; letter-spacing: .03em; }}
      [data-testid="stMetricValue"] {{ font-weight: 700; color: {C_NAVY}; }}
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
      .stTabs [data-baseweb="tab"] {{ background: #fff; border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600; }}
      /* Insight */
      .insight {{ background: #fff; border-left: 5px solid {C_CHURN}; border-radius: 10px;
                  padding: 12px 18px; margin-bottom: 8px; border-top: 1px solid #E6E9F2;
                  border-right: 1px solid #E6E9F2; border-bottom: 1px solid #E6E9F2; font-size: .93rem; }}
      .section-note {{ color: {C_MUTED}; font-size: .85rem; margin: -6px 0 8px; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Churn_Numeric" not in df.columns:
        df["Churn_Numeric"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Senior"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=TENURE_ORDER, ordered=True
    )
    df["ChargeBand"] = pd.cut(
        df["MonthlyCharges"], bins=[0, 35, 55, 75, 95, 1000], labels=CHARGE_ORDER, right=False, ordered=True
    )
    return df


@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame | None:
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


raw = load_default()
if raw is None:
    st.warning("File `telco_churn_cleaned.csv` tidak ditemukan di folder aplikasi. Unggah file di bawah ini.")
    up = st.file_uploader("Upload telco_churn_cleaned.csv", type="csv")
    if up is None:
        st.stop()
    raw = pd.read_csv(up)

df = prepare(raw)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SLICER
# ──────────────────────────────────────────────────────────────────────────────
FILTER_KEYS = ["f_contract", "f_internet", "f_payment", "f_tenure", "f_charge",
               "f_senior", "f_partner", "f_dep", "f_gender"]


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def opts(col):
    return sorted(df[col].dropna().unique().tolist())


with st.sidebar:
    st.markdown("### 🎛️ Filter Segmen")
    st.caption("Semua KPI & grafik mengikuti filter ini.")

    f_contract = st.multiselect("Jenis kontrak", CONTRACT_ORDER, default=CONTRACT_ORDER, key="f_contract")
    f_internet = st.multiselect("Layanan internet", opts("InternetService"),
                                default=opts("InternetService"), key="f_internet")
    f_payment = st.multiselect("Metode pembayaran", opts("PaymentMethod"),
                               default=opts("PaymentMethod"), key="f_payment")

    t_min, t_max = int(df.tenure.min()), int(df.tenure.max())
    f_tenure = st.slider("Masa berlangganan (bulan)", t_min, t_max, (t_min, t_max), key="f_tenure")

    c_min, c_max = float(df.MonthlyCharges.min()), float(df.MonthlyCharges.max())
    f_charge = st.slider("Biaya bulanan ($)", float(np.floor(c_min)), float(np.ceil(c_max)),
                         (float(np.floor(c_min)), float(np.ceil(c_max))), step=1.0, key="f_charge")

    with st.expander("👤 Demografi"):
        f_senior = st.multiselect("Senior citizen", opts("Senior"), default=opts("Senior"), key="f_senior")
        f_partner = st.multiselect("Punya pasangan", opts("Partner"), default=opts("Partner"), key="f_partner")
        f_dep = st.multiselect("Punya tanggungan", opts("Dependents"), default=opts("Dependents"), key="f_dep")
        f_gender = st.multiselect("Gender", opts("gender"), default=opts("gender"), key="f_gender")

    st.button("↺ Reset semua filter", on_click=reset_filters, use_container_width=True)

mask = (
    df.Contract.isin(f_contract)
    & df.InternetService.isin(f_internet)
    & df.PaymentMethod.isin(f_payment)
    & df.tenure.between(*f_tenure)
    & df.MonthlyCharges.between(*f_charge)
    & df.Senior.isin(f_senior)
    & df.Partner.isin(f_partner)
    & df.Dependents.isin(f_dep)
    & df.gender.isin(f_gender)
)
d = df[mask]

with st.sidebar:
    st.divider()
    st.metric("Pelanggan terpilih", f"{len(d):,}", f"{len(d)/len(df):.0%} dari total", delta_color="off")

st.markdown(
    f"""<div class="hero"><h1>📉 Telco Customer Churn Dashboard</h1>
    <p>Memantau siapa yang pergi, seberapa besar pendapatan yang hilang, dan segmen mana yang harus diprioritaskan untuk retensi.</p></div>""",
    unsafe_allow_html=True,
)

if len(d) == 0:
    st.error("Tidak ada pelanggan yang cocok dengan kombinasi filter. Longgarkan filter atau klik Reset.")
    st.stop()
if len(d) < 100:
    st.warning(f"Hanya {len(d)} pelanggan terpilih — hasil kurang stabil secara statistik, tafsirkan dengan hati-hati.")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER ANALITIK & VISUAL
# ──────────────────────────────────────────────────────────────────────────────
def style(fig, height=340, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=C_NAVY),
        title=dict(font=dict(size=14, color=C_NAVY), x=0.01, xanchor="left"),
        showlegend=legend,
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E9ECF3", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def seg_stats(data, col, order=None):
    g = (data.groupby(col, observed=True)
         .agg(n=("Churn_Numeric", "size"), churned=("Churn_Numeric", "sum"),
              mrr_lost=("MonthlyCharges", lambda s: s[data.loc[s.index, "Churn_Numeric"] == 1].sum()))
         .reset_index())
    g["rate"] = g.churned / g.n
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    return g


def rate_bar(data, col, title, order=None, horizontal=False, height=330):
    """Bar churn rate; merah bila di atas rata-rata segmen terpilih, abu bila di bawah."""
    g = seg_stats(data, col, order)
    if not order:
        g = g.sort_values("rate", ascending=horizontal)
    avg = data.Churn_Numeric.mean()
    colors = [C_CHURN if r > avg else C_NEUTRAL for r in g.rate]
    cd = np.stack([g.n, g.churned], axis=-1)
    hover = "<b>%{customdata[0]:,}</b> pelanggan<br>%{customdata[1]:,} churn<extra></extra>"
    if horizontal:
        fig = go.Figure(go.Bar(y=g[col].astype(str), x=g.rate, orientation="h", marker_color=colors,
                               text=[pct(r) for r in g.rate], textposition="outside",
                               customdata=cd, hovertemplate=hover))
        fig.add_vline(x=avg, line_dash="dot", line_color=C_MUTED,
                      annotation_text=f"rata-rata {pct(avg)}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%", range=[0, max(g.rate.max() * 1.25, avg * 1.3)])
    else:
        fig = go.Figure(go.Bar(x=g[col].astype(str), y=g.rate, marker_color=colors,
                               text=[pct(r) for r in g.rate], textposition="outside",
                               customdata=cd, hovertemplate=hover))
        fig.add_hline(y=avg, line_dash="dot", line_color=C_MUTED,
                      annotation_text=f"rata-rata {pct(avg)}", annotation_position="top right")
        fig.update_yaxes(tickformat=".0%", range=[0, max(g.rate.max() * 1.25, avg * 1.3)])
    fig.update_layout(title=title, bargap=0.35)
    return style(fig, height)


@st.cache_data(show_spinner=False)
def driver_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CAT_COLS:
        ct = pd.crosstab(data[c], data["Churn"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rates = data.groupby(c)["Churn_Numeric"].mean()
        rows.append({"Variabel": c, "Cramér's V": v, "p-value": p, "Signifikan": p < 0.05,
                     "Selisih churn (pp)": (rates.max() - rates.min()) * 100,
                     "Kategori terburuk": rates.idxmax()})
    return pd.DataFrame(rows).sort_values("Cramér's V", ascending=False).reset_index(drop=True)


def build_insights(data, base):
    out = []
    avg = data.Churn_Numeric.mean()
    for col, label in [("Contract", "Kontrak"), ("InternetService", "Layanan internet")]:
        g = seg_stats(data, col)
        g = g[g.n >= 30]
        if len(g) >= 2:
            w = g.sort_values("rate", ascending=False).iloc[0]
            out.append(f"**{label} <u>{w[col]}</u>** punya churn <b>{pct(w.rate)}</b> "
                       f"({w.rate / avg:.1f}× rata-rata) — {w.n:,} pelanggan, ${w.mrr_lost:,.0f}/bln hilang.")
    churned = data[data.Churn_Numeric == 1]
    if len(churned):
        early = (churned.tenure <= 12).mean()
        out.append(f"<b>{pct(early, 0)}</b> dari seluruh pelanggan yang churn pergi dalam <b>12 bulan pertama</b> "
                   f"→ fokuskan program onboarding.")
    g = seg_stats(data[data.InternetService != "No"], "TechSupport")
    if set(g.TechSupport) >= {"Yes", "No"} and g.n.min() >= 30:
        no = g.loc[g.TechSupport == "No", "rate"].iloc[0]
        yes = g.loc[g.TechSupport == "Yes", "rate"].iloc[0]
        out.append(f"Pelanggan internet <b>tanpa TechSupport</b> churn {pct(no)} vs {pct(yes)} yang berlangganan "
                   f"({no / max(yes, 1e-9):.1f}× lebih tinggi) → peluang up-sell sekaligus retensi.")
    return out[:4]


# ──────────────────────────────────────────────────────────────────────────────
# KPI
# ──────────────────────────────────────────────────────────────────────────────
n, churned = len(d), int(d.Churn_Numeric.sum())
rate, base_rate = churned / n, df.Churn_Numeric.mean()
mrr, mrr_lost = d.MonthlyCharges.sum(), d.loc[d.Churn_Numeric == 1, "MonthlyCharges"].sum()
filtered = len(d) != len(df)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total pelanggan", f"{n:,}")
k2.metric("Pelanggan churn", f"{churned:,}")
k3.metric("Churn rate", pct(rate),
          f"{(rate - base_rate) * 100:+.1f} pp vs semua" if filtered else None, delta_color="inverse")
k4.metric("MRR hilang", f"${mrr_lost:,.0f}", f"{mrr_lost / mrr:.1%} dari MRR segmen", delta_color="off")
k5.metric("Rata-rata tenure churn", f"{d.loc[d.Churn_Numeric == 1, 'tenure'].mean():.0f} bln",
          f"vs {d.loc[d.Churn_Numeric == 0, 'tenure'].mean():.0f} bln yang bertahan", delta_color="off")

st.write("")
for txt in build_insights(d, df):
    st.markdown(f'<div class="insight">{txt}</div>', unsafe_allow_html=True)
st.write("")


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(
    ["📌 Ringkasan", "🛠️ Layanan & Pembayaran", "👥 Profil Pelanggan", "🔎 Faktor Pendorong", "🎯 Segmen Prioritas"]
)

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<p class="section-note">Merah = di atas rata-rata churn segmen terpilih. Garis putus-putus = rata-rata.</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        show(rate_bar(d, "Contract", "Churn rate per jenis kontrak", CONTRACT_ORDER))
    with c2:
        show(rate_bar(d, "InternetService", "Churn rate per layanan internet"))
    with c3:
        show(rate_bar(d, "TenureGroup", "Churn rate per masa berlangganan", TENURE_ORDER))

    c4, c5 = st.columns([3, 2])
    with c4:
        piv = d.pivot_table(index="Contract", columns="TenureGroup", values="Churn_Numeric",
                            aggfunc="mean", observed=False).reindex(CONTRACT_ORDER)
        cnt = d.pivot_table(index="Contract", columns="TenureGroup", values="Churn_Numeric",
                            aggfunc="size", observed=False).reindex(CONTRACT_ORDER).fillna(0)
        z = piv.where(cnt >= 20)  # sembunyikan sel dengan sampel terlalu kecil
        text = [[("n<20" if pd.isna(v) else pct(v, 0)) for v in row] for row in z.values]
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[str(c) for c in z.columns], y=z.index, text=text, texttemplate="%{text}",
            customdata=cnt.values, hovertemplate="%{y} · %{x}<br>Churn %{z:.1%}<br>%{customdata:,.0f} pelanggan<extra></extra>",
            colorscale=[[0, "#FDF3F3"], [0.5, "#F1A5A8"], [1, C_CHURN]], zmin=0, zmax=0.7,
            colorbar=dict(tickformat=".0%", thickness=10, len=0.8)))
        fig.update_layout(title="Titik panas churn: kontrak × masa berlangganan")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        show(style(fig, 320))
    with c5:
        g = seg_stats(d, "Contract", CONTRACT_ORDER)
        fig = go.Figure(go.Bar(x=g.Contract.astype(str), y=g.mrr_lost, marker_color=C_CHURN,
                               text=[f"${v:,.0f}" for v in g.mrr_lost], textposition="outside",
                               hovertemplate="%{x}<br>MRR hilang $%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="MRR hilang per kontrak ($/bulan)", bargap=0.4)
        fig.update_yaxes(range=[0, g.mrr_lost.max() * 1.2 if g.mrr_lost.max() > 0 else 1])
        show(style(fig, 320))

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    c1, c2 = st.columns([3, 2])
    with c1:
        net = d[d.InternetService != "No"]
        rows = []
        for a in ADDONS:
            g = seg_stats(net, a).set_index(a)
            if {"Yes", "No"} <= set(g.index) and g.n.min() >= 20:
                rows.append({"Layanan": a, "Tanpa": g.loc["No", "rate"], "Berlangganan": g.loc["Yes", "rate"],
                             "n_no": g.loc["No", "n"], "n_yes": g.loc["Yes", "n"]})
        if rows:
            ad = pd.DataFrame(rows)
            ad["gap"] = ad.Tanpa - ad.Berlangganan
            ad = ad.sort_values("gap")
            fig = go.Figure()
            for _, r in ad.iterrows():
                fig.add_trace(go.Scatter(x=[r.Berlangganan, r.Tanpa], y=[r.Layanan] * 2, mode="lines",
                                         line=dict(color="#D5DAE6", width=4), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=ad.Tanpa, y=ad.Layanan, mode="markers+text", name="Tidak berlangganan",
                                     marker=dict(color=C_CHURN, size=14), text=[pct(v, 0) for v in ad.Tanpa],
                                     textposition="top center", customdata=ad.n_no,
                                     hovertemplate="%{y} — tanpa<br>Churn %{x:.1%}<br>%{customdata:,} pelanggan<extra></extra>"))
            fig.add_trace(go.Scatter(x=ad.Berlangganan, y=ad.Layanan, mode="markers+text", name="Berlangganan",
                                     marker=dict(color=C_STAY, size=14), text=[pct(v, 0) for v in ad.Berlangganan],
                                     textposition="bottom center", customdata=ad.n_yes,
                                     hovertemplate="%{y} — berlangganan<br>Churn %{x:.1%}<br>%{customdata:,} pelanggan<extra></extra>"))
            fig.update_layout(title="Dampak add-on terhadap churn (hanya pelanggan internet)")
            fig.update_xaxes(tickformat=".0%", gridcolor="#E9ECF3", showgrid=True)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            show(style(fig, 400, legend=True))
        else:
            st.info("Data pelanggan internet pada filter ini terlalu sedikit untuk membandingkan add-on.")
    with c2:
        show(rate_bar(d, "PaymentMethod", "Churn rate per metode pembayaran", horizontal=True, height=400))

    c3, c4, c5 = st.columns(3)
    with c3:
        show(rate_bar(d, "PaperlessBilling", "Paperless billing", height=290))
    with c4:
        show(rate_bar(d, "MultipleLines", "Multiple lines", height=290))
    with c5:
        show(rate_bar(d, "ChargeBand", "Pita biaya bulanan", CHARGE_ORDER, height=290))

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    c1, c2, c3, c4 = st.columns(4)
    for col, lab, box in [("Senior", "Senior citizen", c1), ("Partner", "Punya pasangan", c2),
                          ("Dependents", "Punya tanggungan", c3), ("gender", "Gender", c4)]:
        with box:
            show(rate_bar(d, col, lab, height=280))

    c5, c6 = st.columns(2)
    with c5:
        fig = px.histogram(d, x="tenure", color="Churn", nbins=24, barmode="overlay", histnorm="percent",
                           opacity=0.65, color_discrete_map={"Yes": C_CHURN, "No": C_STAY})
        fig.update_layout(title="Distribusi masa berlangganan (% dalam kelompok)")
        fig.update_xaxes(title="Tenure (bulan)")
        fig.update_yaxes(title=None)
        show(style(fig, 340, legend=True))
    with c6:
        fig = px.violin(d, x="Churn", y="MonthlyCharges", color="Churn", box=True, points=False,
                        category_orders={"Churn": ["No", "Yes"]},
                        color_discrete_map={"Yes": C_CHURN, "No": C_STAY})
        fig.update_layout(title="Sebaran biaya bulanan vs status churn")
        fig.update_yaxes(title="Biaya bulanan ($)")
        show(style(fig, 340))

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with t4:
    drv = driver_table(d)
    c1, c2 = st.columns([3, 2])
    with c1:
        top = drv.head(12).sort_values("Cramér's V")
        fig = go.Figure(go.Bar(
            y=top.Variabel, x=top["Cramér's V"], orientation="h",
            marker_color=[C_CHURN if s else C_NEUTRAL for s in top.Signifikan],
            text=[f"{v:.2f}" for v in top["Cramér's V"]], textposition="outside",
            customdata=np.stack([top["p-value"], top["Selisih churn (pp)"], top["Kategori terburuk"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Selisih churn %{customdata[1]:.1f} pp<br>"
                          "Terburuk: %{customdata[2]}<br>p=%{customdata[0]:.2e}<extra></extra>"))
        fig.update_layout(title="Kekuatan asosiasi variabel dengan churn (Cramér's V · merah = signifikan p<0.05)")
        fig.update_xaxes(range=[0, max(top["Cramér's V"].max() * 1.2, 0.1)])
        show(style(fig, 440))
    with c2:
        nums = ["tenure", "MonthlyCharges", "TotalCharges"]
        cr = pd.Series({c: d[c].corr(d.Churn_Numeric) for c in nums if d[c].std() > 0}).sort_values()
        fig = go.Figure(go.Bar(y=cr.index, x=cr.values, orientation="h",
                               marker_color=[C_CHURN if v > 0 else C_STAY for v in cr.values],
                               text=[f"{v:+.2f}" for v in cr.values], textposition="outside"))
        fig.update_layout(title="Korelasi variabel numerik dengan churn")
        fig.update_xaxes(range=[-0.6, 0.6], zeroline=True, zerolinecolor="#9AA3B5")
        show(style(fig, 440))
        st.caption("Negatif (hijau) = makin besar nilainya, makin kecil peluang churn. "
                   "Korelasi menunjukkan asosiasi, bukan sebab-akibat.")

    st.markdown("##### Jelajahi variabel apa pun")
    var = st.selectbox("Pilih variabel", drv.Variabel.tolist() if len(drv) else CAT_COLS, label_visibility="collapsed")
    order = TENURE_ORDER if var == "TenureGroup" else None
    e1, e2 = st.columns([2, 3])
    with e1:
        show(rate_bar(d, var, f"Churn rate · {var}", order, height=300))
    with e2:
        st.dataframe(
            drv.style.format({"Cramér's V": "{:.3f}", "p-value": "{:.2e}", "Selisih churn (pp)": "{:.1f}"}),
            hide_index=True, height=300, width="stretch")

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<p class="section-note">Segmen = kombinasi kontrak × layanan internet × masa berlangganan. '
                'Urutkan berdasarkan churn rate (siapa paling rentan) atau MRR hilang (siapa paling mahal).</p>',
                unsafe_allow_html=True)
    f1, f2 = st.columns([1, 1])
    min_n = f1.slider("Minimum pelanggan per segmen", 20, 300, 50, step=10)
    sort_by = f2.radio("Urutkan berdasarkan", ["Churn rate", "MRR hilang"], horizontal=True)

    seg = (d.groupby(["Contract", "InternetService", "TenureGroup"], observed=True)
           .apply(lambda g: pd.Series({
               "Pelanggan": len(g),
               "Churn": int(g.Churn_Numeric.sum()),
               "Churn rate (%)": g.Churn_Numeric.mean() * 100,
               "MRR hilang ($)": g.loc[g.Churn_Numeric == 1, "MonthlyCharges"].sum(),
               "Avg biaya bulanan ($)": g.MonthlyCharges.mean()}), include_groups=False)
           .reset_index())
    seg["TenureGroup"] = seg.TenureGroup.astype(str)
    seg = seg[seg["Pelanggan"] >= min_n]
    seg = seg.sort_values("Churn rate (%)" if sort_by == "Churn rate" else "MRR hilang ($)",
                          ascending=False).reset_index(drop=True)

    if seg.empty:
        st.info("Tidak ada segmen yang memenuhi minimum pelanggan. Turunkan batas minimum.")
    else:
        top5 = seg.head(5).copy()
        top5["Segmen"] = top5.Contract + " · " + top5.InternetService + " · " + top5.TenureGroup
        fig = go.Figure(go.Bar(
            y=top5.Segmen[::-1], x=top5["Churn rate (%)"][::-1] / 100, orientation="h", marker_color=C_CHURN,
            text=[f"{r:.0f}%  ·  ${m:,.0f}/bln" for r, m in zip(top5["Churn rate (%)"][::-1], top5["MRR hilang ($)"][::-1])],
            textposition="outside"))
        fig.update_layout(title=f"Top 5 segmen berisiko (urut {sort_by.lower()}) — label: churn rate · MRR hilang")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.0])
        show(style(fig, 300))

        st.dataframe(
            seg, hide_index=True, width="stretch",
            column_config={
                "Churn rate (%)": st.column_config.ProgressColumn("Churn rate", format="%.1f%%", min_value=0, max_value=100),
                "MRR hilang ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Avg biaya bulanan ($)": st.column_config.NumberColumn(format="$%.2f"),
            })
        st.download_button("⬇️ Unduh daftar segmen (CSV)", seg.to_csv(index=False).encode(),
                           "segmen_prioritas_churn.csv", "text/csv")

st.caption(f"Sumber: telco_churn_cleaned.csv · {len(df):,} pelanggan · MRR = jumlah MonthlyCharges.")
