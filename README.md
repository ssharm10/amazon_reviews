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

### 🚀 Key Phases

1. **Data Collection:** The foundation of any data-driven project is access to high-quality data. For this project, we acquire the necessary data from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw) especially focusing on the Handmade items category. This dataset spans product categories for Handmade items (over 600k reviews, 400MB metadata), making it ideal for studying product popularity and review credibility. This dataset includes rich metadata and customer reviews, providing a comprehensive foundation for analysis.
Amazon Review 2023 is an updated version of the Amazon Review 2018 dataset. This dataset mainly includes reviews (ratings, text) and item metadata (descriptions, category information, price, brand, and images). Compared to the previous versions, the 2023 version features larger size, newer reviews (up to Sep 2023), richer and cleaner meta data, and finer-grained timestamps.

2. **[Data Loading and Cleaning](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/01-data-loading-cleaning_subsampled.ipynb)**: In this notebook, we work with a sample of the dataset to develop and test our preprocessing pipeline. We load the meta and review datasets, clean them, and merge them to create a unified dataset for exploration and modeling. This step is critical to ensure the accuracy and reliability of our results. Key tasks include making columns atomic (e.g unpacking details or categories column), cleaning redundanct columns and addressing inconsistencies in the data. Once the pipeline is validated, we plan to apply the same techniques to the entire dataset.

3. **[Prelimnary EDA](https://github.com/ssharm10/amazon_reviews/blob/main/notebooks/02-eda_subsampled.ipynb)**: In this notebook, we perform an in-depth analysis of the dataset to uncover trends, relationships, and patterns. We use visualizations and statistical summaries to gain insights into factors influencing product ratings.

### Data Dictionary

| Column Name                | Data Type     | Description                                                                                   |
|----------------------------|---------------|-----------------------------------------------------------------------------------------------|
| `rating_by_user`       | int64          | Rating given by the user. |
| `title_review`         | object         | Title of the user's review. |
| `text_review`         | object         | Full text of the user's review. |
| `product_id`          | object         | Unique identifier for the reviewed product. |
| `user_id`             | object         | Unique identifier for the user. |
| `time_of_review`      | datetime64[ns] | Timestamp when the review was written. |
| `helpful_vote`        | int64          | Number of users who found the review helpful. |
| `verified_purchase`   | bool           | Whether the user purchased the product. |                                            |
| `title_product`                   | object        | Name of the product.                                                                          |
| `average_rating`          | float64       | Rating of the product on the product page.                                                    |
| `date_first_available`    | datetime64[ns]| Date when the product was first available for purchase.   
| `rating_number`           | int64         | Number of ratings for the product.                                                            |
| `features`                | object        | List of features of the product.                                                  |
| `description`             | object        | Description of the product.                                                                   |
| `price`                   | float64       | Price in US dollars                                                  |
| `store`                   | object        | Store name where the product is sold.                                                         |                                                    |
| `parent_asin`             | object        | Parent ID of the product.                                                                     |
| `sub_category_1`          | object        | First sub-category of the product.                                                            |
| `sub_category_2`          | object        | Second sub-category of the product.                                                           |
| `sub_category_3`          | object        | Third sub-category of the product.                                                            |                                    |
| `combined_category`          | object        | Combined category of the product.                                                            |                                    |
| `package_dimensions_inches`   | object              | Package dimensions in inches.                                                                                                               |
| `package_weight_ounces`       | float64             | Package weight in ounces.                                                                                                              |
| `package_length_inches`       | float64             | Package length in inches.                                                                                                              |
| `package_width_inches`        | float64             | Package width in inches.                                                                                                               |
| `package_height_inches`       | float64             | Package height in inches.                                                                                                              |
| `department`         | object              | Cleaned department names for the product.                                                                                             |
---