import streamlit as st 
import pandas as pd
from enhanced_recommender import get_recommendations
from tokenizer import custom_tokenizer

# --- Custom CSS Styling ---
st.markdown("""
<style>
/* Sidebar width */
[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}

/* Sidebar button styling */
[data-testid="stSidebar"] .stButton > button {
    border: none !important;
    background: none !important;
    text-align: left !important;
    padding: 0.5rem 0.75rem !important;
    width: 100% !important;
    font-size: 1rem !important;
    color: #333 !important;
    transition: color 0.2s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
}

[data-testid="stSidebar"] .stButton > button:hover {
    color: #e75480 !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

.active-btn {
    font-weight: bold !important;
    color: #e75480 !important;
}

/* Slider styling */
.stSlider .st-eb {
    background-color: #e75480 !important;
}

/* Button styling */
.stButton>button {
    background-color: #e75480 !important;
    color: white !important;
    border: none !important;
    padding: 0.7rem 1.4rem !important;
    font-size: 1.1rem !important;
    border-radius: 10px !important;
    transition: background-color 0.3s ease !important;
    margin: 0 auto !important;
    display: block !important;
    width: auto !important;
}

.stButton>button:hover {
    background-color: #d74372 !important;
}

/* Center content */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation Logic ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Welcome"

pages = {
    "🏠 Welcome": "Welcome",
    "🛒 Product Recommender": "Product Recommender",
    "📞 Contact": "Contact"
}

with st.sidebar:
    st.markdown('<div style="font-size:1.3rem; font-weight:600; margin-bottom:1rem;">📚 Navigation</div>', unsafe_allow_html=True)
    for label, key in pages.items():
        btn_class = "active-btn" if st.session_state.page == label else ""
        button_html = f'<span class="{btn_class}">{label}</span>'
        if st.button(label, key=key):
            st.session_state.page = label
            st.rerun()