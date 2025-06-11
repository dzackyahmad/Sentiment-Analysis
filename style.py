import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* ========== GLOBAL RESET ========== */
    html, body {
        background-color: #121212;
        color: #f1f1f1;
        font-family: 'Segoe UI', sans-serif;
        scroll-behavior: smooth;
    }

    /* ========== HEADINGS ========== */
    h1, h2, h3 {
        color: #ffc107;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* ========== BUTTON ========== */
    .stButton>button {
        background: linear-gradient(135deg, #ffc107, #ff9800);
        color: black;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 12px;
        transition: 0.3s ease;
        box-shadow: 0 0 10px rgba(255,193,7,0.4);
    }
    .stButton>button:hover {
        background: #fff176;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255,193,7,0.6);
    }

    /* ========== INPUTS ========== */
    .stTextArea textarea, .stTextInput input, .stSelectbox div {
        background-color: #1e1e1e !important;
        color: #f1f1f1 !important;
    }

    /* ========== METRIC + BAR CHARTS ========== */
    .stMetricValue {
        color: #ffc107 !important;
    }
    .stPlotlyChart .bar {
        fill: #ffc107 !important;
    }

    /* ========== SIDEBAR ========== */
    .css-1d391kg { background-color: #181818 !important; }
    .css-h5rgaw, .css-1v3fvcr {
        color: #ffc107 !important;
    }

    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        color: #ffc107 !important;
        font-weight: bold;
    }

    /* ========== ANIMATED LINK (FOOTER/INFO) ========== */
    a {
        color: #ffeb3b;
        text-decoration: none;
        transition: 0.2s ease;
    }
    a:hover {
        color: #ffffff;
        text-shadow: 0 0 10px #ffeb3b;
    }

    /* ========== CUSTOM SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1c1c1c;
    }
    ::-webkit-scrollbar-thumb {
        background: #ffc107;
        border-radius: 6px;
    }

    /* ========== CARD-LIKE SHADOW ========== */
    .stTextArea, .stButton, .stMetric, .stDataFrame, .stSelectbox {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
