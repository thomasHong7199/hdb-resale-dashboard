import pandas as pd
import streamlit as st
import os

st.set_page_config(page_title="HDB Resale – Data Check", layout="wide")

CSV_PATH = "data/resale_all_years.csv"   # update file name if needed

@st.cache_data
def get_data():
    # Load CSV safely
    if not os.path.exists(CSV_PATH):
        st.error(f"File not found at {CSV_PATH}")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Load once
df = get_data()
# --- KPIs ---
st.subheader("📊 Key Metrics")

# --- Add price per sqm column early ---
if "price_psm" not in df.columns and "floor_area_sqm" in df.columns:
    df["price_psm"] = df["resale_price"] / df["floor_area_sqm"]

# --- Now show metrics ---
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Price (SGD)", f"{df['resale_price'].mean():,.0f}")

with col2:
    import numpy as np
    median_psm = df["price_psm"].median() if "price_psm" in df.columns else None
    st.metric(
        "Median Price / sqm",
        f"{median_psm:,.0f}" if median_psm is not None and not np.isnan(median_psm) else "N/A"
    )

with col3:
    st.metric("Highest Transaction", f"{df['resale_price'].max():,.0f}")

with col4:
    st.metric("Total Transactions", f"{len(df):,}")


# Show preview
st.title("HDB Resale Dataset Preview")
st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
st.dataframe(df.head(20))
import numpy as np

# 1) Normalize column names (safe to run even if already normalized)
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_").str.replace("-", "_"))

# 2) If your file used a different name, map it to price_psm
aliases = ["price_per_sqm", "price_sqm", "price_per_sq_m"]
for a in aliases:
    if a in df.columns and "price_psm" not in df.columns:
        df = df.rename(columns={a: "price_psm"})

# 3) If still missing, compute it from price and floor area
if "price_psm" not in df.columns:
    if {"resale_price", "floor_area_sqm"}.issubset(df.columns):
        # ensure numeric and avoid divide-by-zero
        df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
        df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
        df["price_psm"] = df["resale_price"] / df["floor_area_sqm"]
        df.loc[~np.isfinite(df["price_psm"]), "price_psm"] = np.nan  # inf/0 → NaN
    else:
        st.warning("Missing columns to compute price_psm (need resale_price and floor_area_sqm).")

# 4) Make month a proper datetime (helps the trend chart)
if "month" in df.columns and not np.issubdtype(df["month"].dtype, np.datetime64):
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

import plotly.express as px

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filters")
if "month" in df.columns:
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    years = sorted(df["month"].dt.year.dropna().unique().tolist())
else:
    years = []

year_pick = st.sidebar.selectbox("Year", ["All"] + years, index=0)
town_pick = st.sidebar.multiselect("Town(s)", sorted(df["town"].dropna().unique().tolist()) if "town" in df else [])
flat_pick = st.sidebar.multiselect("Flat type(s)", sorted(df["flat_type"].dropna().unique().tolist()) if "flat_type" in df else [])

# filtered data
dff = df.copy()
if year_pick != "All" and "month" in dff:
    dff = dff[dff["month"].dt.year == year_pick]
if town_pick:
    dff = dff[dff["town"].isin(town_pick)]
if flat_pick:
    dff = dff[dff["flat_type"].isin(flat_pick)]

# ----------------- FILTERED KPIs -----------------
st.subheader("📊 Key Metrics (Filtered)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Price (SGD)", f"{dff['resale_price'].mean():,.0f}")
c2.metric("Median $/sqm", f"{dff['price_psm'].median():,.0f}" if "price_psm" in dff else "N/A")
c3.metric("Highest Transaction", f"{dff['resale_price'].max():,.0f}")
c4.metric("Total Transactions", f"{len(dff):,}")

# ----------------- 1) PRICE TREND -----------------
st.subheader("📈 Average Resale Price Over Time")
if "month" in dff:
    trend = (dff.groupby("month", as_index=False)["resale_price"].mean()
               .sort_values("month"))
    fig_trend = px.line(trend, x="month", y="resale_price",
                        labels={"month":"Month","resale_price":"Avg Price (SGD)"})
    st.plotly_chart(fig_trend, use_container_width=True, key="chart_trend")
else:
    st.info("No 'month' column to plot the trend.")

# ----------------- 2) FLAT TYPE DONUT -----------------
st.subheader("🏠 Market Share by Flat Type (Total Value)")
if "flat_type" in dff:
    share = (dff.groupby("flat_type", as_index=False)["resale_price"].sum()
               .sort_values("resale_price", ascending=False))
    fig_donut = px.pie(share, names="flat_type", values="resale_price", hole=0.45)
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True, key="chart_donut")
else:
    st.info("No 'flat_type' column for donut chart.")

# ----------------- 3) TOP TOWNS BAR -----------------
st.subheader("🏙️ Top Towns by Avg $/sqm")
if {"town","price_psm"}.issubset(dff.columns):
    by_town = (dff.groupby("town", as_index=False)["price_psm"].mean()
                 .sort_values("price_psm", ascending=False)
                 .head(20))
    fig_town = px.bar(by_town, x="price_psm", y="town",
                      labels={"price_psm":"Avg $/sqm","town":"Town"})
    st.plotly_chart(fig_town, use_container_width=True, key="chart_town")
else:
    st.info("Need 'town' and 'price_psm' to plot town comparison.")
