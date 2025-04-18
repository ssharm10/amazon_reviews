import pandas as pd
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import unicodedata
import logging
import streamlit as st
logging.basicConfig(level=logging.INFO)


# Then Load the large English pipeline
nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])  # Disabling parser & NER for efficiency
logging.info("spaCy loaded successfully!")  # Logging 

# Cache the vectorizer to avoid recomputing
@st.cache_resource(ttl=3600)
def load_tfidf():
    """Load the pre-fitted vectorizer and matrix"""
    vectorizer = joblib.load('./Streamlit/data/tfidf_vectorizer.joblib')
    tfidf_matrix = load_npz('./Streamlit/data/tfidf_matrix.npz')
    return tfidf_matrix, vectorizer


#Function to generate recommendations
def get_recommendations(df, item_title, top_n=8, text_weight=0.7, 
                                          numeric_weights={'bayesian_rating': 0.7, 
                                                          'product_price': -0.3},
                                                          rating_threshold = 20,
                                                          new_product_threshold=1500):
    """
    Enhanced content-based recommendation system that combines text similarity with weighted numeric features, 
    and ensures a mix of atleast one new product with popular ones.
    
    Parameters:
    - df: DataFrame containing product information
    - item_title: Title of the item to find recommendations for
    - top_n: Number of recommendations to return
    - text_weight: Weight for text similarity (0-1)
    - numeric_weights: Dictionary of weights for numeric features (must sum to 1)
    - new_product_threshold: Age in days below which a product is considered "new"
    
    Returns:
    - DataFrame with top_n recommended items and their details
    """
  
    # Extract the parent_asin of the item 
    product_id = df.loc[df['product_title'] == item_title,'parent_asin'].values[0]
    item_index = df[df['parent_asin'] == product_id].index[0]
    

    # Load pre-computed TF-IDF matrix and fitted vectorizer
    tfidf_matrix, vectorizer = load_tfidf()

    # Calculate cosine similarity only for the target item
    query_vec = vectorizer.transform([df.loc[item_index, 'title_category']])
    text_sim = cosine_similarity(query_vec, tfidf_matrix).flatten() 
    
    #save as a new column in the dataframe
    df['text_similarity'] = text_sim

    #  Normalize numerical features (0 to 1)
    df_normalized = df.copy()
    for col in numeric_weights:
        if numeric_weights[col] > 0:  # Higher is better
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:  # Lower is better (e.g., price)
            df_normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Calculate numerical scores
    df['numeric_score'] = (
    df_normalized['bayesian_rating'] * 0.7 -  # Recommend popular products
    df_normalized['product_price'] * 0.3  # Recommend less pricey products
)
    
    # Combine scores
    df['combined_score'] = (text_weight * df['text_similarity']) + ((1 - text_weight) * df['numeric_score'])
    
    # Save results into dataframe
    sim_df = pd.DataFrame({
        'product_title': df['product_title'],
        'similarity_score': df['combined_score'].round(2),
        'bayesian_rating': df['bayesian_rating'].round(2),
        'rating_number': df['rating_number'],
        'product_age_days': df['product_age_days'],
    })
    logging.info(f"sim_df shape: {sim_df.shape}")
    # Sorting similar items by similarity score and rating number in descending order
    similar_items = sim_df.sort_values(by=["similarity_score","rating_number"],ascending=False)

    # Exclude the item itself
    similar_items = similar_items.loc[similar_items['product_title'] != item_title]

    # Filter items that meet the rating threshold
    qualified_items = similar_items[similar_items["rating_number"] > rating_threshold]

    # Identify new products (below new_product_threshold)
    new_items = qualified_items[qualified_items['product_age_days'] <= new_product_threshold]
    logging.info(f"new_items length: {len(new_items)}")
    # Ensure at least 1 new product is included if possible
    if len(new_items) >= 1:
        new_items = new_items.head(1)  # Pick the top new item
    else:
        new_items = pd.DataFrame()  # No new products, continue with popular items

    # Select popular items to fill the remaining spots
    popular_items = qualified_items.head(top_n - len(new_items))

    # Combine both new items and popular items
    top_similar_items = pd.concat([new_items, popular_items])
    logging.info(f"top_similar_items shape: {top_similar_items.shape}")
    # Re-sort the combined results by similarity_score  and rating_number
    top_similar_items = top_similar_items.sort_values(
    by=["similarity_score", "rating_number"],
    ascending=[False, False]
    ).head(top_n)

    return top_similar_items
