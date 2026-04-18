import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="CSV Data Dashboard", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    h1, h2, h3 { color: #1a5276; }
    .stDownloadButton>button { background-color: #1a5276; color: white; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 CSV Data Visualization Dashboard")
st.caption("Upload your CSV file or generate a sample dataset to explore visualizations and statistics.")
st.divider()

with st.sidebar:
    st.header("📁 Dataset Source")
    data_source = st.radio("Choose data source", ["Upload CSV", "Generate Random Data"])
    if data_source == "Generate Random Data":
        n_rows = st.slider("Number of rows", 50, 500, 100)
        n_cols = st.slider("Number of columns", 2, 8, 4)
    st.divider()
    st.header("🎨 Chart Settings")
    chart_color = st.color_picker("Chart color", "#1a5276")
    show_grid = st.checkbox("Show grid on charts", value=True)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: **{uploaded_file.name}** — {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
else:
    np.random.seed(42)
    col_names = [f"Feature_{chr(65+i)}" for i in range(n_cols)]
    data = np.random.randn(n_rows, n_cols).cumsum(axis=0)
    df = pd.DataFrame(data, columns=col_names)
    st.info(f"🎲 Generated random dataset — {n_rows} rows × {n_cols} columns")

if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("📌 Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(numeric_cols))
    col4.metric("Missing Values", int(df.isnull().sum().sum()))
    st.divider()

    with st.expander("🔍 Show Dataset Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe().round(3), use_container_width=True)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        with st.expander("⚠️ Missing Values Report"):
            st.dataframe(missing.rename("Missing Count").reset_index(), use_container_width=True)
            if st.button("Fill missing values with column mean"):
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.success("Missing values filled with column means.")

    st.divider()
    st.subheader("📊 Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Line Chart", "📊 Histogram", "🔥 Correlation Heatmap", "📦 Box Plot"])

    with tab1:
        selected = st.multiselect("Select columns to plot", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        if selected:
            fig, ax = plt.subplots(figsize=(10, 4))
            for col in selected:
                ax.plot(df[col], label=col, linewidth=1.8)
            ax.set_title("Line Chart", fontsize=14, fontweight='bold', color='#1a5276')
            ax.legend()
            if show_grid: ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig); plt.close()

    with tab2:
        hist_col = st.selectbox("Select column for histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 60, 20)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[hist_col].dropna(), bins=bins, color=chart_color, edgecolor='white', alpha=0.85)
        ax.set_title(f"Histogram — {hist_col}", fontsize=14, fontweight='bold', color='#1a5276')
        if show_grid: ax.grid(True, alpha=0.3, axis='y')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with tab3:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols)-1)))
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(corr.columns, fontsize=9)
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                            fontsize=8, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
            ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold', color='#1a5276', pad=15)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")

    with tab4:
        box_cols = st.multiselect("Select columns for box plot", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        if box_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            bp = ax.boxplot([df[c].dropna().values for c in box_cols], patch_artist=True, labels=box_cols)
            for patch in bp['boxes']:
                patch.set_facecolor(chart_color); patch.set_alpha(0.7)
            ax.set_title("Box Plot — Distribution Overview", fontsize=14, fontweight='bold', color='#1a5276')
            if show_grid: ax.grid(True, alpha=0.3, axis='y')
            ax.spines[['top', 'right']].set_visible(False)
            plt.xticks(rotation=30, ha='right'); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("⬇️ Download Processed Data")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(label="Download CSV", data=csv_buffer.getvalue(), file_name="processed_data.csv", mime="text/csv")

else:
    st.info("👈 Upload a CSV file or generate random data from the sidebar to get started.")

st.divider()
st.caption("Built by Sourabh Saw | Python • Pandas • Streamlit • Matplotlib")
