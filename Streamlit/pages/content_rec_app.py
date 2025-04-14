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

# --- Load Data ---
@st.cache_resource
def load_data():
    return pd.read_pickle("./Streamlit/data/content_rec_data.pkl")

rec_data = load_data()

# --- Main Page Inputs ---
    
st.markdown("""
<div style="text-align: center;">
<h4>Find Your Perfect Match</h4>
</div>
""", unsafe_allow_html=True)
st.subheader("3 Simple Steps:")
st.markdown("""
1. **🔍 Search** - Type or select a product below  
2. **📏 Customize** - Choose how many recommendations you want (1-20)  
3. **⭐ Filter** - Set a minimum rating count for quality assurance (20-1000) 
""")

item_title = st.selectbox(
    "Select a Product:", 
    rec_data["product_title"],
    help="Start typing to search products"
)
top_n = st.slider("Number of recommendations:", 1, 20, 8)
rating_threshold = st.slider("Minimum ratings:", 20, 1000, 20)

# Check if inputs have changed
if (item_title != st.session_state.item_title or 
    top_n != st.session_state.top_n or
    rating_threshold != st.session_state.rating_threshold):
    st.session_state.run_recommender = False   

# Button triggers recommendation generation (Styled & Centered)
col1, col2, col3 = st.columns([2, 1, 2])  # Centering the button
with col2:
    if st.button("Recommend", key="recommend-btn", use_container_width=False):
        st.session_state["run_recommender"] = True
        st.session_state["item_title"] = item_title
        st.session_state["top_n"] = top_n
        st.session_state["rating_threshold"] = rating_threshold