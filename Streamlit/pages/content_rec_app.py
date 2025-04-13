import streamlit as st
import pandas as pd
from enhanced_recommender import get_recommendations

# --- Apply Custom CSS for Sidebar Buttons ---
st.markdown(
    """
    <style>
    /* Style for sidebar buttons */
    .stButton>button {
        background-color: #ff4b4b !important;  /* Red color */
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
        width: 100% !important;
    }
    
    /* Style for the Recommend button (centered & smaller width) */
    .recommend-btn {
        display: flex;
        justify-content: center;
    }
    .recommend-btn button {
        background-color: #f63366 !important; /* Streamlit default primary color */
        color: white !important;
        border-radius: 10px !important;
        width: auto !important;
        padding: 10px 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navigation Sidebar ---
with st.sidebar:
    st.markdown("## Navigation")

    pages = {
        "🏠 Welcome": "welcome",
        "🛒 Product Recommender": "recommender",
        "📞 Contact": "contact"
    }

    for label, page_key in pages.items():
        if st.button(label, key=page_key):
            st.session_state.page = page_key

# --- Product Recommender Page ---
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# Display content based on the selected page
if st.session_state.page == 'welcome':
    # --- Welcome Page Layout ---
    st.title("🛍️ Welcome to the Amazon Pro Recommender 🛍️")
    
    st.markdown("""  

    This **next-gen recommender** combines signals from different product attributes to surface truly relevant products on Amazon.
                
    ### Key Features of the Recommender:
    - **🔍 Textual Similarity**: Finds hidden gems using NLP on product titles/categories
    - **📈 Confidence-Boosted Ratings**: Helps prioritize truly popular products using Bayesian ratings.
    - **📊 Business Logic**: Factors in pricing and rating counts.
    - **🆕 New Product Visibility**: Prioritizes at least one new product to give newer items a chance to gain traction.
    
    **Why this works?**
    - Solves the "cold start" problem - Recommends great products immediately, even for new items with no purchase history.
    - Works without mass user data - Delivers personalized suggestions where most customers only buy once.
    - Fraud-resistant design - Bayesian ratings automatically downweight suspicious products (like those with few inflated ratings) while promoting genuinely popular items.
                
    *See it in action → Click "Product Recommender" in the sidebar*
    """)

elif st.session_state.page == "recommender":
    st.markdown("<h1 style='text-align: center;'>Amazon Pro Recommender</h1>", 
                unsafe_allow_html=True)

    # Initialize session state variables
    if "run_recommender" not in st.session_state:
        st.session_state.run_recommender = False
    if "item_title" not in st.session_state:
        st.session_state.item_title = ""
    if "top_n" not in st.session_state:
        st.session_state.top_n = 8
    if "rating_threshold" not in st.session_state:
        st.session_state.rating_threshold = 20

    # Cache data loading
    @st.cache_data
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

    # --- Display Recommendations ---
    if st.session_state.get("run_recommender", False): 
        with st.spinner("Hang on tight, Generating recommendations..."):
            recommendations = get_recommendations(
                rec_data, 
                st.session_state["item_title"], 
                top_n=st.session_state["top_n"],
                rating_threshold=st.session_state["rating_threshold"]
            )
        
        st.success("Done!")
        
        if recommendations.empty:
            st.warning("No recommendations found. Try relaxing your filters (e.g., lower rating threshold).")
        else:
            st.subheader(f"✨ Recommendations for: {st.session_state['item_title']}")
            st.dataframe(recommendations)

elif st.session_state.page == "contact":
    st.title("Let's Chat!")
    
    st.markdown("""
    **Hi there! I'm Soniya**  
    *Data Scientist • Former NASA Researcher • Science Storyteller*
    """)
    
    st.markdown("""
    My data science journey began among the stars (literally!) - from reconstructing galaxies at NASA to 
    optimizing chip fabrication at Intel. When I realized how much I loved turning complex data into 
    actionable insights, I took the leap into data science through BrainStation's intensive bootcamp.
    """)
    
    st.markdown("""
    This recommender system is just one slice of my capstone project! On [GitHub](github.com/ssharm10), 
    you can explore:
    - 🛍️ **How I predicted Amazon product popularity** using machine learning
    - 📊 **The full analysis** from metadata cleaning to Bayesian ratings
    - 🧠 **Why I chose this hybrid approach** over traditional methods
    """)
    
    st.markdown("""
    What excites me most? Building solutions that actually work in the real world!
    """)
    
    st.markdown("---")
    
    st.markdown("**Let's talk data, ML, or even astrophysics!**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("✉️ [soniya.iiser@gmail.com]")  
        st.markdown("🔗 [linkedin.com/in/sharma-soniya]")  
    with col2:
        st.markdown("💻 [github.com/ssharm10]")  
    
    st.markdown("*P.S. Ask me about my FameLab science communication experience!*")