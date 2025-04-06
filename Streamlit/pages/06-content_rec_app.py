import streamlit as st
import pandas as pd
from enhanced_recommender import get_recommendations

# Configure warnings and page settings
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Amazon Recommender Pro",
    page_icon="🛍️",
    layout="wide"
)

# # Initialize session state variables
# if 'run_recommender' not in st.session_state:
#     st.session_state.run_recommender = False
# if 'item_title' not in st.session_state:
#     st.session_state.item_title = ""
# if 'top_n' not in st.session_state:
#     st.session_state.top_n = 8
# if 'rating_threshold' not in st.session_state:
#     st.session_state.rating_threshold = 20

# Cache data loading
@st.cache_data
def load_data():
    return pd.read_pickle('../data/content_rec_data.pkl')

rec_data = load_data()

# --- Main Page Layout ---
st.title("🔍 Amazon Product Recommender")
st.markdown("""
    **Discover Similar Products** with Bayesian Ratings and Content-Based Filtering.

    *How it works:*  
    - Enter a product name  
    - Adjust filters (ratings, number of recs)  
    - Click Recommend button to get results  
    
    ### Key Features of the Recommender:
    - **Text Based Similarity**: Calculates Cosine Similarity using both the product titles and categories.
    - **Bayesian Ratings**: Helps prioritize truly popular products.
    - **Rating Number**: Boosts products with higher number of ratings.
    - **Product Price**: Recommends lower-priced products where appropriate.
    - **New Product Visibility**: Prioritizes at least one new product to give newer items a chance to gain traction.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Settings")
    item_title = st.selectbox(
        "Select a Product:", 
        rec_data['product_title'],
        help="Start typing to search products"
    )
    top_n = st.slider("Number of recommendations:", 1, 20, 8)
    rating_threshold = st.slider("Minimum ratings:", 0, 1000, 20)

    # Check if inputs have changed
    if (item_title != st.session_state.item_title or 
        top_n != st.session_state.top_n or
        rating_threshold != st.session_state.rating_threshold):
        st.session_state.run_recommender = False   

    # Button triggers recommendation generation
    if st.button("Recommend", use_container_width=True):
        st.session_state['run_recommender'] = True
        st.session_state['item_title'] = item_title
        st.session_state['top_n'] = top_n
        st.session_state['rating_threshold'] = rating_threshold

# --- Display Recommendations on Main Page ---
if st.session_state.get("run_recommender", False): 
    with st.spinner('Hang on tight, Generating recommendations...'):
        recommendations = get_recommendations(
            rec_data, 
            st.session_state['item_title'], 
            top_n=st.session_state['top_n'],
            rating_threshold=st.session_state['rating_threshold']
        )
    
    st.success("Done!")
    
    if recommendations.empty:
        st.warning("No recommendations found. Try relaxing your filters (e.g., lower rating threshold).")
    else:
        st.subheader(f"✨ Recommendations for: {st.session_state['item_title']}")
        st.dataframe(recommendations)