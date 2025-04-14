''' 
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
@st.cache_data
def load_data():
    return pd.read_pickle("../../Streamlit/data/content_rec_data.pkl")

rec_data = load_data()

# --- Welcome Page ---
if st.session_state.page == "🏠 Welcome":
    st.markdown("""
    <h1 style='text-align: center; font-size: 1.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
        🛍️ Welcome to the Amazon Pro Recommender 🛍️
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""  
    *This next-gen recommender combines signals from different product attributes to surface truly relevant products on Amazon.*
                
    #### Key Features of the Recommender:
    - **🔍 Textual Similarity**: Finds hidden gems using NLP on product titles/categories
    - **📈 Confidence-Boosted Ratings**: Helps prioritize truly popular products using Bayesian ratings.
    - **📊 Business Logic**: Factors in pricing and rating counts.
    - **🆕 New Product Visibility**: Prioritizes at least one new product to give newer items a chance to gain traction.
    
    #### 💡 Why this Works:

    - **Works Without Mass User Data**  
    Delivers personalized suggestions by analyzing product content — ideal for platforms where users often make one-off purchases and traditional collaborative filtering falls short.

    - **Solves the "Cold Start" Problem for Users**  
    Because it's content-based, it can recommend similar items even if a user has only interacted with one product — no need for long histories or lots of reviews.

    - **Minimizes Review Manipulation Impact**  
    Bayesian adjustment and minimum rating thresholds ensure products are fairly ranked, protecting against spammy or biased reviews.

    - **Elevates Relevant Hidden Gems**  
    NLP-based similarity highlights lesser-known but closely related items, surfacing options beyond just bestsellers.

    - **Business Logic Built In**  
    The system blends ratings, price, and visibility for new products — providing practical, balanced results for e-commerce scenarios.
                
    ##### *See it in action → Click "Product Recommender" in the sidebar*
    """)

# --- Recommender Page ---
elif st.session_state.page == "🛒 Product Recommender":
    st.markdown("<h1 style='text-align: center;'>Amazon Pro Recommender</h1>", 
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center;">
    <h4>🛍️ Find Your Perfect Match 🛍️ </h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #f9f9f9; border-left: 5px solid #e75480; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; margin-bottom: 1.5rem;">
            <h4 style="margin-top: 0;">3 Simple Steps:</h4>
            <ul style="padding-left: 1.2rem; margin-bottom: 0;">
                <li><b>🔍 Search</b> – Type or select a product below</li>
                <li><b>📏 Customize</b> – Choose how many recommendations you want (1–20)</li>
                <li><b>⭐ Filter</b> – Set a minimum rating count for quality assurance (20–1000)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    item_title = st.selectbox(
        "Select a Product:", 
        rec_data["product_title"],
        help="Start typing to search products"
    )
    top_n = st.slider("Number of recommendations:", 1, 20, 8)
    rating_threshold = st.slider("Minimum ratings:", 20, 1000, 20)

    # Centered recommend button
    if st.button("Recommend", key="recommend-btn"):
        with st.spinner("Hang on tight, generating recommendations..."):
            recommendations = get_recommendations(
                rec_data, 
                item_title, 
                top_n=top_n,
                rating_threshold=rating_threshold
            )

        st.success("Done!")

        if recommendations.empty:
            st.warning("No recommendations found. Try relaxing your filters (e.g., lower rating threshold).")
        else:
            st.subheader(f"✨ Recommendations for: {item_title}")
            st.dataframe(recommendations)

# --- Contact Page ---
elif st.session_state.page == "📞 Contact":
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
    This recommender system is just one slice of my capstone project! On [GitHub](https://github.com/ssharm10), 
    you can explore:
    - 🛍️ **How I predicted Amazon product popularity**
    - 📊 **The full analysis** from metadata cleaning to Bayesian ratings
    - 🧠 **Why I chose this hybrid approach**
    """)

    st.markdown("""
    What excites me most? Building solutions that actually work in the real world!
    """)

    st.markdown("---")
    st.markdown("**Let's talk data, ML, or even astrophysics!**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("✉️ [soniya.iiser@gmail.com](mailto:soniya.iiser@gmail.com)")  
        st.markdown("🔗 [LinkedIn](https://linkedin.com/in/sharma-soniya)")  
    with col2:
        st.markdown("💻 [GitHub](https://github.com/ssharm10)")
'''
import streamlit as st
import pandas as pd
from enhanced_recommender import get_recommendations
from tokenizer import custom_tokenizer

# # --- Apply Custom CSS for Sidebar Buttons ---
# st.markdown(
#     """
#     <style>
#     /* Style for sidebar buttons */
#     .stButton>button {
#         background-color: #ff4b4b !important;  /* Red color */
#         color: white !important;
#         border-radius: 5px !important;
#         border: none !important;
#         width: 100% !important;
#     }
    
#     /* Style for the Recommend button (centered & smaller width) */
#     .recommend-btn {
#         display: flex;
#         justify-content: center;
#     }
#     .recommend-btn button {
#         background-color: #f63366 !important; /* Streamlit default primary color */
#         color: white !important;
#         border-radius: 10px !important;
#         width: auto !important;
#         padding: 10px 20px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

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
 