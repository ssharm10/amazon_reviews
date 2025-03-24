## Predicting Product Popularity
=========================
## Table of Contents

## Table of Contents

1. [Predicting Product Popularity](#predicting-product-popularity)
2. [Project Overview](#project-overview)
   - [Motivation](#motivation)
   - [Potential Impact](#potential-impact)
   - [Challenges](#challenges)
3. [Proposed Data Science Solution](#proposed-data-science-solution)
4. [Key Phases](#-key-phases)
   - [Data Collection](#data-collection)
   - [Data Loading and Cleaning](#data-loading-and-cleaning)
   - [Preliminary EDA](#preliminary-eda)
   - [Pre-processing and detailed EDA](#preprocessing)
   - [Merge meta data and reviews](#merge)
   - [Baseline Modeling](#baseline-model)
5. [Data Dictionary](#data-dictionary)


### Project Overview
- **Motivation**<br>
My area of interest lies in e-commerce and retail analytics, with a focus on predicting product reception and customer sentiment analysis. A key challenge in this domain is accurately estimating the rating a new product will receive before its launch. Traditional methods rely heavily on post-launch sales data and customer feedback, which limits the ability to make proactive decisions. By leveraging rich metadata—such as product category, price, brand, and features—this project aims to provide early insights into consumer perception and address this critical problem. Accurate predictions of product reception are not only vital for optimizing marketing and pricing strategies but also for preventing significant financial losses, as businesses lose $1.1 trillion annually globally due to inventory shortages or overstocking.

<br>

- **Potential Impact**<br>
This project has far reaching benefits across the ecosystem.
    - **Consumers** – More accurate product ratings and helpful reviews reduce misleading information, leading to better purchasing decisions and fewer returns.
    - **Retailers & Brands** – Predicting product reception helps optimize inventory, refine marketing strategies, and protect brand reputation from review manipulation.
    - **E-Commerce Platforms** – Enhancing review credibility ensures users see the most relevant feedback, improving trust and engagement on the platform.

<br>

- **Challenges**<br>
This project integrates structured metadata analysis and unstructured text processing from the Amazon 2023 dataset to address key challenges in e-commerce analytics, including:
    -	extracting insights from diverse metadata
    -	handling review quality variability – some reviews contain more useful insights than others
    -	addressing data imbalance – some products have higher reviews than others <br>

Despite these challenges, the impact is significant—better demand forecasting, improved recommendations, and greater consumer trust, making this a valuable area for innovation. 

<br>

### Proposed Data Science Solution<br>

This project proposes developing a predictive model** to estimate the expected rating of a product based on its metadata and historical review data. By leveraging machine learning techniques, the model will analyze patterns in product features, customer feedback, and other relevant factors to provide early insights into product reception. This solution aims to empower businesses with actionable predictions, enabling better decision-making in inventory planning, pricing strategies, and marketing efforts, while reducing risks associated with poor demand forecasting.

### 🚀 Methodology

1. **Data Collection:** The foundation of any data-driven project is access to high-quality data. For this project, we acquire the necessary data from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw) especially focusing on the Handmade items category. This dataset spans product categories for Handmade items (over 600k reviews, 400MB metadata), making it ideal for studying product popularity and review credibility. This dataset includes rich metadata and customer reviews, providing a comprehensive foundation for analysis.
Amazon Review 2023 is an updated version of the Amazon Review 2018 dataset. This dataset mainly includes reviews (ratings, text) and item metadata (descriptions, category information, price, brand, and images). Compared to the previous versions, the 2023 version features larger size, newer reviews (up to Sep 2023), richer and cleaner meta data, and finer-grained timestamps.

2. **[Data Loading and Cleaning](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/01-data-loading-cleaning_subsampled.ipynb)**: In this notebook, we work with a sample of the dataset to develop and test our preprocessing pipeline. We load the meta and review datasets, clean them, and merge them to create a unified dataset for exploration and modeling. This step is critical to ensure the accuracy and reliability of our results. Key tasks include making columns atomic (e.g unpacking details or categories column), cleaning redundanct columns and addressing inconsistencies in the data. Once the pipeline is validated, we plan to apply the same techniques to the entire dataset.

3. **[Prelimnary EDA](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/03-eda_subsampled.ipynb)**: In this notebook, we perform an in-depth analysis of the dataset to uncover trends, relationships, and patterns. We use visualizations and statistical summaries to gain insights into factors influencing product ratings.

4. **[Pre Processing & Detailed EDA](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/03-preprocessing_meta.ipynb)**: This notebook focuses on further preprocessing of columns in the meta data including handling missing values, cleaning stores and subcategories. We also conduct detailed EDA to visualize relationships between our features and the target variable. Additionaly, we explore feature engineering to create more meaningful and interpretable features suitable for modeling.

5. **[Merging Reviews and Meta](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/04-meta_review_merge.ipynb)**: In this notebook, we merge the reviews dataset with meta data after extracting store-based and category-based statistics of user reviews from the reiews dataset. This step enriches the metadata with aggregated insights from user reviews, enabling a more comprehensive analysis and modeling process.

6. **[Baseline Model](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/05-baseline_modeling.ipynb)**: This notebook focuses on building and evaluating two baseline machine learning models- Logistic Regression and Random Forest.

### Data Dictionary


| #  | Column                                              | Data Type  | Description |
|----|-----------------------------------------------------|------------|-------------|
| 0  | product_price                                      | float64    | Price of the product |
| 1  | highly_rated_product                              | int64      | Binary flag (1 = highly rated, 0 = not highly rated) |
| 2  | product_title_length                              | int64      | Length of the product title (number of characters) |
| 3  | store_grouped_total_products                      | int64      | Total number of products sold by the store |
| 4  | average_rating                                    | float64    | Average rating of the product |
| 5  | num_product_images                                | int64      | Number of images associated with the product |
| 6  | store_grouped_weighted_mean_rating               | float64    | Weighted mean rating of all products in the store |
| 7  | store_grouped_std_rating                         | float64    | Standard deviation of product ratings in the store |
| 8  | store_grouped_low_rated_ratio                    | float64    | Ratio of low-rated products in the store |
| 9  | has_package_density                              | int64      | Binary flag (1 = product has package density, 0 = no) |
| 10 | package_density                                  | float64    | Density of the product packaging |
| 11 | product_age_days                                 | int64      | Number of days since the product was first listed |
| 12 | store_grouped                                   | object     | Store name (grouped by common identifier) |
| 13 | parent_asin                                     | object     | Parent ASIN (Amazon Standard Identification Number) |
| 14 | subcategory1_total_products                     | int64      | Total number of products in subcategory 1 |
| 15 | subcategory1_mean_rating                        | float64    | Mean rating of products in subcategory 1 |
| 16 | subcategory1_std_rating                         | float64    | Standard deviation of ratings in subcategory 1 |
| 17 | subcategory1_std_rating_number                  | float64    | Standard deviation of rating count in subcategory 1 |
| 18 | combined_category_weighted_mean_rating         | float64    | Weighted mean rating of all products in the combined category |
| 19 | combined_category_total_products               | int64      | Total number of products in the combined category |
| 20 | combined_category_mean_rating_number           | float64    | Mean number of ratings per product in the combined category |
| 21 | combined_category_std_rating_number            | float64    | Standard deviation of number of ratings in the combined category |
| 22 | combined_category_low_rated_ratio              | float64    | Ratio of low-rated products in the combined category |
| 23 | spacy_tokenized_features                       | object     | Tokenized product features using spaCy |
| 24 | combined_category                              | object     | Category name after combining multiple levels |
| 25 | num_reviews_store_grouped                      | int64      | Total number of reviews for the store |
| 26 | avg_user_rating_store_grouped                  | float64    | Average user rating for the store |
| 27 | std_user_rating_store_grouped                  | float64    | Standard deviation of user ratings in the store |
| 28 | review_title_word_counts_store_grouped         | float64    | Average word count of review titles in the store |
| 29 | review_text_word_counts_store_grouped          | float64    | Average word count of review texts in the store |
| 30 | verified_purchase_ratio_store_grouped          | float64    | Ratio of verified purchases in the store |
| 31 | weighted_helpfulness_store_grouped             | float64    | Weighted helpfulness score for reviews in the store |
| 32 | one_to_five_star_store_grouped                 | float64    | Ratio of 1-star to 5-star reviews in the store |
| 33 | num_reviews_combined_category_grouped          | int64      | Total number of reviews in the combined category |
| 34 | avg_user_rating_combined_category_grouped      | float64    | Average user rating in the combined category |
| 35 | std_user_rating_combined_category_grouped      | float64    | Standard deviation of user ratings in the combined category |
| 36 | review_title_word_counts_combined_category_grouped  | float64 | Average word count of review titles in the combined category |
| 37 | review_text_word_counts_combined_category_grouped   | float64 | Average word count of review texts in the combined category |
| 38 | one_to_five_star_combined_category_grouped     | float64    | Ratio of 1-star to 5-star reviews in the combined category |