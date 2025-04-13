import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import unicodedata


# # First download the Large English Pipeline from Spacy
#!python -m spacy download en_core_web_lg

# Then Load the large English pipeline
nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])  # Disabling parser & NER for efficiency

def normalize_text(text):
    """
    Normalizes text by converting special Unicode characters into standard ASCII.
    """
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return normalized_text

#Custom tokenizer to vectorize textual features
def custom_tokenizer(row):
    """
    Tokenizes and lemmatizes a product title, normalizing Unicode characters
    and removing stopwords.

    Args:
        row (str): A product title or description.

    Returns:
        str: Processed text with lemmatized words.
    """
    # Normalize Unicode styles
    normalized_text = normalize_text(row)

    # Process text with SpaCy
    parsed_title = nlp(normalized_text)

    # Extract only relevant tokens
    tok_lemmas = [
        token.lemma_.lower()    # Convert lemma to lowercase
        for token in parsed_title 
        if token.is_alpha       # Ensure token is alphabetic
        and not token.is_stop   # Remove stopwords
        and len(token) > 3      # Ignore very short words
    ]

    # Remove duplicates while preserving order
    unique_tokens = list(dict.fromkeys(tok_lemmas))

    return unique_tokens  # Convert list to a single string

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

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=10,
        max_df=0.7,
        stop_words='english'
    )

    # Extract the parent_asin of the item 
    product_id = df.loc[df['product_title'] == item_title,'parent_asin'].values[0]
    item_index = df[df['parent_asin'] == product_id].index[0]

    # Apply TF-IDF vectorization to text features
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['title_category'])
    
    # Calculate cosine similarity for text features
    text_sim = cosine_similarity(tfidf_matrix)
    
    df['text_similarity'] = text_sim[item_index]

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

    # Sorting similar items by similarity score and rating number in descending order
    similar_items = sim_df.sort_values(by=["similarity_score","rating_number"],ascending=False)

    # Exclude the item itself
    similar_items = similar_items.loc[similar_items['product_title'] != item_title]

    # Filter items that meet the rating threshold
    qualified_items = similar_items[similar_items["rating_number"] > rating_threshold]

    # Identify new products (below new_product_threshold)
    new_items = qualified_items[qualified_items['product_age_days'] <= new_product_threshold]

    # Ensure at least 1 new product is included if possible
    if len(new_items) >= 1:
        new_items = new_items.head(1)  # Pick the top new item
    else:
        new_items = pd.DataFrame()  # No new products, continue with popular items

    # Select popular items to fill the remaining spots
    popular_items = qualified_items.head(top_n - len(new_items))

    # Combine both new items and popular items
    top_similar_items = pd.concat([new_items, popular_items]).head(top_n)

    return top_similar_items