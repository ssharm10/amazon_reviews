import streamlit as st
import pandas as pd
import os
from enhanced_recommender import get_recommendations

# Configure warnings and page settings
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Amazon Recommender Pro",
    page_icon="🛍️",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        .big-font {
            font-size: 40px !important;
            color: #3E6B8B;
            font-weight: bold;
        }
        .intro-text {
            font-size: 18px;
            color: #4C4C4C;
        }
        .custom-btn {
            background-color: #FF7F50;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 16px;
            text-align: center;
        }
        .header { 
            background-color: #FF7F50;
            padding: 20px;
            border-radius: 10px;
            color: white;
            font-size: 28px;
            font-weight: bold;
        }
        .sidebar {
            background-color: #F7F7F7;
            padding: 20px;
            border-radius: 10px;
        }
        .main-title {
            font-size: 36px;
            color: #FF7F50;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #FF7F50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize the session state if it's not already initialized
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Sidebar navigation with buttons
with st.sidebar:
    st.markdown("## Navigation")
    
    pages = {
        '🏠 Welcome': 'welcome',
        '🛒 Product Recommender': 'recommender',
        'Contact': 'contact'
    }

    # Create navigation buttons
    for label, page_key in pages.items():
        if st.button(label, key=page_key, use_container_width=True, type="primary" if st.session_state.page == page_key else "secondary"):
            st.session_state.page = page_key
            st.rerun()


# Display content based on the selected page
if st.session_state.page == 'welcome':
    # --- Welcome Page Layout ---
    st.title("🛍️ Welcome to the Amazon Pro Recommender 🛍️")
    
    st.markdown("""
    ## Project Overview:
    <p class="list-style">
        In the competitive world of e-commerce, enhancing the customer experience is key to driving sales, improving customer retention, and building long-term relationships. This project seeks to tackle some of the most pressing challenges in the field by developing a state-of-the-art product recommender system. The aim is to provide personalized and relevant product recommendations based on customer preferences, behavior, and a combination of advanced recommendation techniques.
    </p>
    
    <p class="list-style">
        The core of this project is a <b><span class="highlight">hybrid recommender system</span></b> that integrates a variety of methods to deliver more precise and diverse recommendations. This system is designed not only to predict which products customers are most likely to purchase but also to surface products that match their unique tastes and preferences. Here's how the model works:
    </p>
    
    <ul class="list-style">
        <li>🔍 <b><span class="highlight">Textual Similarity</span></b>: The system leverages natural language processing (NLP) to analyze product titles, descriptions, and categories, helping to identify products that share similar attributes and characteristics. This helps customers discover items that align with their specific needs.</li>
        <li>📈 <b><span class="highlight">Bayesian Ratings</span></b>: To refine the accuracy of recommendations, Bayesian ratings adjust product scores based on factors such as confidence and historical performance. Products that are frequently rated highly are prioritized, while new or less-popular items are given a fair chance.</li>
        <li>📊 <b><span class="highlight">Business Logic Integration</span></b>: The recommender model incorporates various business rules, including price sensitivity (e.g., suggesting products within a customer’s budget) and the number of ratings (to avoid recommending products with insufficient feedback). These rules add an important layer of contextual relevance.</li>
        <li>🆕 <b><span class="highlight">New Product Visibility</span></b>: A key feature of this system is ensuring that new products are not left out. The model actively includes newly launched products in the recommendations, helping to give them visibility and potentially accelerate their adoption.</li>
        <li>🔄 <b><span class="highlight">Hybrid Approach</span></b>: By combining textual similarity, Bayesian ratings, and business logic, the recommender system offers a balanced approach that caters to both popular, highly-rated products and less-known, emerging items that customers may find interesting.</li>
    </ul>

    <p class="list-style">
        Overall, this hybrid recommendation engine is designed to enhance the customer shopping experience by ensuring that recommendations are not only relevant but also dynamic and varied. By offering personalized suggestions and boosting product visibility, we aim to increase sales and drive customer loyalty.
    </p>
""", unsafe_allow_html=True)
    
    # st.markdown("""
    # ## Project Overview:
    # This project tackles critical challenges in e-commerce:

    # Personalized recommendations drive repeat purchases, improving customer retention and sales growth.
    
    # We build a hybrid recommender model combining:

    # - **Textual similarity** (NLP on product titles/categories)
    # - **Bayesian ratings** (confidence-weighted scores)
    # - **Business logic** (price sensitivity, rating counts)
    # - **New Product Visibility**: Ensures recommendations include at least one new product
    # This recommendation engine balances textual similarity with Bayesian-adjusted ratings to surface high-potential products.

    # """)

elif st.session_state.page == 'recommender':
    # --- Product Recommender Page Layout ---
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

    # Initialize session state variables
    if 'run_recommender' not in st.session_state:
        st.session_state.run_recommender = False
    if 'item_title' not in st.session_state:
        st.session_state.item_title = ""
    if 'top_n' not in st.session_state:
        st.session_state.top_n = 8
    if 'rating_threshold' not in st.session_state:
        st.session_state.rating_threshold = 20

    # Cache data loading
    @st.cache_data
    def load_data():
        # Print the files in the directory
        files_in_directory = os.listdir('../data')
        st.write("Files in data directory:", files_in_directory)
        return pd.read_pickle('../data/content_rec_data.pkl')

    rec_data = load_data()

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