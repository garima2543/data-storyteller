"""Data Storyteller — Streamlit Dashboard (fully offline, zero API dependency)"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_storyteller import (
    validate_df, missing_summary, top_correlations, generate_insights
)

st.set_page_config(
    page_title="Data Storyteller",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #faf9f6; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="metric-container"] { background: #f2f0eb; border-radius: 12px; padding: 16px 20px; border: none; }
[data-testid="metric-container"] > div { gap: 4px; }
[data-testid="stMetricLabel"] > div { font-size: 11px !important; font-weight: 700 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #9a9894 !important; }
[data-testid="stMetricValue"] > div { font-family: 'DM Serif Display', serif !important; font-size: 28px !important; color: #1a1916 !important; }
[data-testid="stExpander"] { border: 1px solid rgba(26,25,22,0.1) !important; border-radius: 12px !important; background: #fff !important; }
.stButton > button { background: #1a1916 !important; color: #faf9f6 !important; border: none !important; border-radius: 8px !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important; font-size: 14px !important; padding: 10px 24px !important; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
hr { border-color: rgba(26,25,22,0.1) !important; }
[data-testid="stSidebar"] { background: #f2f0eb !important; }
</style>
""", unsafe_allow_html=True)

AMBER        = "#c97a1a"
TEAL         = "#1a8a6e"
CORAL        = "#c94a2a"
BLUE         = "#1a5ac9"
PURPLE       = "#6a3ac9"
CHART_COLORS = [AMBER, TEAL, CORAL, BLUE, PURPLE, "#8a5ac9", "#5ac9a9", "#c95a8a"]
TAG_ICONS    = {"warn": "⚠️", "success": "✅", "info": "ℹ️", "alert": "🔺"}
TAG_COLORS   = {"warn": "#fef3dc", "success": "#e0f5ee", "info": "#e8effc", "alert": "#fceee8"}
TAG_BORDER   = {"warn": "#f0c86a", "success": "#7dd4b8", "info": "#99bbe8", "alert": "#e8a08a"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#5a5955"),
    margin=dict(l=0, r=0, t=24, b=0),
    xaxis=dict(gridcolor="#e8e5de", linecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="#e8e5de", linecolor="rgba(0,0,0,0)"),
)


@st.cache_data
def demo_dataframe():
    np.random.seed(42)
    n = 300
    cats     = ["Electronics", "Clothing", "Books", "Home", "Sports"]
    regions  = ["North", "South", "East", "West", "Central"]
    statuses = ["Completed", "Pending", "Returned", "Cancelled"]
    qty   = np.random.randint(1, 11, n)
    price = np.round(np.random.uniform(10, 400, n), 2)
    return pd.DataFrame({
        "order_id":      range(1001, 1001 + n),
        "category":      [cats[i % len(cats)] for i in range(n)],
        "region":        [regions[i % len(regions)] for i in range(n)],
        "status":        [statuses[i % len(statuses)] for i in range(n)],
        "quantity":      qty,
        "unit_price":    price,
        "total_revenue": np.round(qty * price, 2),
        "discount_pct":  np.round(np.random.uniform(0, 30, n), 1),
        "customer_age":  np.random.randint(18, 68, n),
        "rating":        np.round(np.random.uniform(2, 5, n), 1),
        "delivery_days": np.random.randint(1, 15, n),
        "return_flag":   np.where(np.random.random(n) > 0.85, 1, 0),
    })


def fmt(v):
    v = float(v)
    if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
    return f"{v:,.2f}"


def narrative_box(text):
    st.markdown(
        f'''<div style="background:#1a1916;border-radius:14px;padding:28px 32px;margin-bottom:8px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-40px;right:-40px;width:160px;height:160px;border:40px solid rgba(255,255,255,0.04);border-radius:50%;"></div>
  <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{AMBER};margin-bottom:14px;">── DATA NARRATIVE</div>
  <div style="font-size:16px;line-height:1.8;color:rgba(250,249,246,0.85);">{text}</div>
</div>''', unsafe_allow_html=True)


def insight_card(tag, text, num):
    st.markdown(
        f'''<div style="background:{TAG_COLORS[tag]};border:1px solid {TAG_BORDER[tag]};border-radius:12px;padding:18px 20px;margin-bottom:12px;position:relative;overflow:hidden;">
  <span style="position:absolute;top:8px;right:14px;font-family:'DM Serif Display',serif;font-size:44px;color:rgba(0,0,0,0.07);line-height:1;">{num:02d}</span>
  <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;color:#444;">{TAG_ICONS[tag]} {tag.upper()}</div>
  <div style="font-size:14px;line-height:1.6;color:#1a1916;">{text}</div>
</div>''', unsafe_allow_html=True)


def corr_badge(val):
    val = float(val)
    if val > 0.7:
        return f'<span style="background:#fceee8;color:{CORAL};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{val:.2f}</span>'
    if val > 0.4:
        return f'<span style="background:#fef3dc;color:{AMBER};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{val:.2f}</span>'
    return f'<span style="background:#f2f0eb;color:#9a9894;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{val:.2f}</span>'


def build_narrative(df):
    try:
        numeric = df.select_dtypes(include=[np.number])
        cat     = df.select_dtypes(exclude=[np.number])
        ms      = missing_summary(df)
        total_miss   = int(ms["missing_count"].sum())
        total_cells  = len(df) * df.shape[1]
        completeness = round((1 - total_miss / total_cells) * 100, 1) if total_cells > 0 else 100.0
        high_miss    = ms[ms["missing_pct"] > 10]
        parts = []
        parts.append(
            f"This dataset contains {len(df):,} records across {df.shape[1]} fields "
            f"({numeric.shape[1]} numeric, {cat.shape[1]} categorical) with {completeness}% completeness."
        )
        if not high_miss.empty:
            flagged = ", ".join(f"{col} ({row['missing_pct']}%)" for col, row in high_miss.iterrows())
            parts.append(f"Data quality needs attention: {flagged} {'have' if len(high_miss) > 1 else 'has'} notable missing values.")
        else:
            parts.append("Data quality looks healthy — no column exceeds 10% missing values.")
        if not numeric.empty:
            desc = numeric.describe().T
            top_mean_col = desc["mean"].sort_values(ascending=False).index[0]
            top_mean_val = float(desc.loc[top_mean_col, "mean"])
            top_std_col  = desc["std"].sort_values(ascending=False).index[0]
            top_std_val  = float(desc.loc[top_std_col, "std"])
            parts.append(
                f"The highest-valued metric by average is {top_mean_col} ({fmt(top_mean_val)}), "
                f"while {top_std_col} shows the most spread (std {fmt(top_std_val)})."
            )
        corr_df = top_correlations(df, n=3)
        if not corr_df.empty:
            row      = corr_df.iloc[0]
            f1       = str(row["feature_1"])
            f2       = str(row["feature_2"])
            corr_val = float(row["corr_val"])
            strength = "very strongly" if corr_val > 0.8 else "moderately" if corr_val > 0.5 else "weakly"
            parts.append(f"Strongest relationship: {f1} and {f2} (|r| = {corr_val:.2f}), they {strength} move together.")
        if not cat.empty:
            col     = cat.columns[0]
            top_val = str(df[col].value_counts().index[0])
            top_pct = round(int(df[col].value_counts().iloc[0]) / len(df) * 100, 1)
            parts.append(f'In "{col}", the dominant value is "{top_val}" ({top_pct}% of records).')
        return "  ".join(parts)
    except Exception:
        return f"Dataset loaded: {len(df):,} rows x {df.shape[1]} columns. Explore the tabs below for analysis."


def chart_numeric(df, metric="mean"):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        st.info("No numeric columns.")
        return
    desc = numeric.describe().T[[metric]].reset_index()
    desc.columns = ["Column", metric]
    fig = px.bar(desc, x="Column", y=metric, color="Column",
                 color_discrete_sequence=CHART_COLORS, text=desc[metric].apply(fmt))
    fig.update_traces(textposition="outside", marker_line_width=0, marker_cornerradius=6)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)


def chart_missing(df):
    ms = missing_summary(df)
    ms = ms[ms["missing_pct"] > 0].head(15)
    if ms.empty:
        st.success("No missing values detected")
        return
    fig = px.bar(ms.reset_index(), x="missing_pct", y="index", orientation="h",
                 color="missing_pct",
                 color_continuous_scale=[[0, BLUE], [0.4, AMBER], [1, CORAL]],
                 text=ms["missing_pct"].apply(lambda v: f"{v}%"))
    fig.update_traces(textposition="outside", marker_line_width=0, marker_cornerradius=4)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False,
                      height=max(220, len(ms) * 36 + 60), yaxis_title="", xaxis_title="Missing %")
    st.plotly_chart(fig, use_container_width=True)


def chart_correlations(df):
    corr_df = top_correlations(df, n=8)
    if corr_df.empty:
        st.info("Not enough numeric columns.")
        return
    corr_df = corr_df.copy()
    corr_df["pair"] = corr_df["feature_1"].astype(str) + " ↔ " + corr_df["feature_2"].astype(str)
    corr_df["corr_val"] = corr_df["corr_val"].astype(float)
    fig = px.bar(corr_df, x="corr_val", y="pair", orientation="h",
                 color="corr_val", color_continuous_scale=[[0, BLUE], [0.5, AMBER], [1, CORAL]],
                 text=corr_df["corr_val"].apply(lambda v: f"{v:.2f}"), range_x=[0, 1])
    fig.update_traces(textposition="outside", marker_line_width=0, marker_cornerradius=4)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False,
                      height=max(220, len(corr_df) * 40 + 60), yaxis_title="", xaxis_title="|Correlation|")
    st.plotly_chart(fig, use_container_width=True)


def chart_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        st.info("Need at least 2 numeric columns.")
        return
    corr = numeric.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, CORAL], [0.5, "#faf9f6"], [1, TEAL]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}", textfont={"size": 10},
        showscale=True,
    ))
    # Build layout without xaxis/yaxis from PLOTLY_LAYOUT to avoid duplicate key error
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    layout["height"] = 420
    layout["xaxis"] = dict(tickangle=-35, gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    layout["yaxis"] = dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


def chart_categorical(df, col):
    counts = df[col].value_counts().head(12).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", color=col,
                 color_discrete_sequence=CHART_COLORS, text="count")
    fig.update_traces(textposition="outside", marker_line_width=0, marker_cornerradius=6)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def chart_scatter(df, x_col, y_col, color_col=None):
    kwargs = dict(x=x_col, y=y_col, opacity=0.65, color_discrete_sequence=[AMBER])
    if color_col and color_col != "(none)":
        kwargs["color"] = color_col
        kwargs["color_discrete_sequence"] = CHART_COLORS
    fig = px.scatter(df, **kwargs)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown("""
    <div style="padding:40px 0 8px;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#c97a1a;margin-bottom:12px;">Drop your CSV. Get the story.</div>
      <div style="font-family:'DM Serif Display',serif;font-size:48px;line-height:1.1;letter-spacing:-1px;color:#1a1916;margin-bottom:12px;">Turn raw data into<br><em style="color:#c97a1a;">clear narratives</em></div>
      <div style="font-size:16px;color:#5a5955;max-width:520px;line-height:1.6;">Upload any ecommerce CSV and instantly see summaries, correlations, missing-value charts, and an auto-generated narrative.</div>
    </div>
    <hr style="margin:28px 0;">
    """, unsafe_allow_html=True)

    col_up, col_demo = st.columns([3, 1], gap="large")
    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    with col_demo:
        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        use_demo = st.button("📂 Load demo dataset")

    df, fname = None, ""
    if use_demo:
        df, fname = demo_dataframe(), "demo_ecommerce.csv"
        st.session_state["df"] = df
        st.session_state["fname"] = fname
        st.session_state.pop("narrative", None)
    elif uploaded:
        df = pd.read_csv(uploaded)
        fname = uploaded.name
        st.session_state["df"] = df
        st.session_state["fname"] = fname
        st.session_state.pop("narrative", None)
    elif "df" in st.session_state:
        df = st.session_state["df"]
        fname = st.session_state.get("fname", "dataset.csv")

    if df is None:
        st.markdown("""
        <div style="background:#f2f0eb;border:1.5px dashed rgba(26,25,22,0.2);border-radius:12px;padding:48px;text-align:center;margin-top:16px;">
          <div style="font-size:32px;margin-bottom:12px;">📂</div>
          <div style="font-size:15px;color:#5a5955;">Upload a CSV file above, or click <strong>Load demo dataset</strong>.</div>
        </div>""", unsafe_allow_html=True)
        return

    valid, msgs = validate_df(df)
    if not valid:
        for m in msgs:
            st.warning(m)
    else:
        st.success(f"Loaded {fname} — {len(df):,} rows x {df.shape[1]} columns")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    numeric = df.select_dtypes(include=[np.number])
    cat     = df.select_dtypes(exclude=[np.number])
    ms_all  = missing_summary(df)
    total_m = int(ms_all["missing_count"].sum())
    comp_pct = round((1 - total_m / (len(df) * df.shape[1])) * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",         f"{len(df):,}")
    c2.metric("Columns",      df.shape[1])
    c3.metric("Numeric",      numeric.shape[1])
    c4.metric("Categorical",  cat.shape[1])
    c5.metric("Completeness", f"{comp_pct}%",
              delta="good" if comp_pct >= 90 else "needs attention",
              delta_color="normal" if comp_pct >= 90 else "inverse")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("### 📖 Data Story")
    if "narrative" not in st.session_state:
        st.session_state["narrative"] = build_narrative(df)
    narrative_box(st.session_state["narrative"])
    if st.button("↻ Refresh narrative"):
        st.session_state["narrative"] = build_narrative(df)
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Auto Insights")
    insights = generate_insights(df)
    col_a, col_b = st.columns(2)
    for i, (tag, text) in enumerate(insights):
        with (col_a if i % 2 == 0 else col_b):
            insight_card(tag, text, i + 1)

    st.markdown("<hr style='margin:28px 0'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Numeric Summary", "🔴 Missing Values",
        "🔗 Correlations", "🏷️ Categorical", "🔬 Explorer",
    ])

    with tab1:
        st.markdown("#### Numeric columns overview")
        metric_choice = st.radio("Metric", ["mean", "median", "std", "min", "max"],
                                 horizontal=True, label_visibility="collapsed")
        chart_numeric(df, metric_choice)
        with st.expander("Full descriptive statistics"):
            st.dataframe(numeric.describe().T.style.format("{:.3f}"), use_container_width=True)

    with tab2:
        st.markdown("#### Missing values by column")
        chart_missing(df)
        with st.expander("Detailed missing data table"):
            missing_data = ms_all[ms_all["missing_count"] > 0]
            if missing_data.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(missing_data.style.format({"missing_pct": "{:.1f}%"}), use_container_width=True)

    with tab3:
        st.markdown("#### Top correlations between numeric columns")
        col_bar, col_heat = st.columns(2, gap="large")
        with col_bar:
            st.markdown("**Ranked pairs**")
            chart_correlations(df)
        with col_heat:
            st.markdown("**Full correlation matrix**")
            chart_heatmap(df)
        with st.expander("Correlation table"):
            corr_df = top_correlations(df)
            if corr_df.empty:
                st.info("Not enough numeric columns.")
            else:
                html_rows = "".join(
                    "<tr>"
                    f"<td style='padding:8px 12px;font-size:12px;font-family:monospace'>{str(row['feature_1'])}</td>"
                    "<td style='padding:8px;color:#9a9894;text-align:center'>↔</td>"
                    f"<td style='padding:8px 12px;font-size:12px;font-family:monospace'>{str(row['feature_2'])}</td>"
                    f"<td style='padding:8px 12px;text-align:right'>{corr_badge(float(row['corr_val']))}</td>"
                    "</tr>"
                    for _, row in corr_df.iterrows()
                )
                st.markdown(
                    "<table style='width:100%;border-collapse:collapse'><thead><tr>"
                    "<th style='text-align:left;font-size:10px;text-transform:uppercase;color:#9a9894;padding:8px 12px;border-bottom:1px solid #e8e5de'>Feature 1</th>"
                    "<th></th>"
                    "<th style='text-align:left;font-size:10px;text-transform:uppercase;color:#9a9894;padding:8px 12px;border-bottom:1px solid #e8e5de'>Feature 2</th>"
                    "<th style='text-align:right;font-size:10px;text-transform:uppercase;color:#9a9894;padding:8px 12px;border-bottom:1px solid #e8e5de'>|r|</th>"
                    f"</tr></thead><tbody>{html_rows}</tbody></table>",
                    unsafe_allow_html=True,
                )

    with tab4:
        if cat.empty:
            st.info("No categorical columns detected.")
        else:
            cat_col = st.selectbox("Select column", cat.columns.tolist())
            chart_categorical(df, cat_col)
            top_vals = df[cat_col].value_counts().head(10).reset_index()
            top_vals.columns = [cat_col, "count"]
            top_vals["share %"] = (top_vals["count"] / len(df) * 100).round(1)
            st.dataframe(top_vals, use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("#### Scatter explorer")
        num_cols = numeric.columns.tolist()
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                x_col = st.selectbox("X axis", num_cols, index=0)
            with ec2:
                y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
            with ec3:
                color_col = st.selectbox("Colour by", ["(none)"] + cat.columns.tolist())
            chart_scatter(df, x_col, y_col, color_col)
        st.markdown("#### Raw data preview")
        n_rows = st.slider("Rows to show", 5, min(200, len(df)), 20)
        st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("""
    <hr style="margin:40px 0 16px">
    <div style="text-align:center;font-size:12px;color:#9a9894;">
      Data Storyteller · Runs entirely in your browser · No data leaves your machine
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()