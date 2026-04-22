# Customer Segmentation with Sentiment Analysis

## 1. Project Overview

This project investigates whether integrating sentiment analysis into customer segmentation can improve clustering performance. The model combines numerical features (Income, Spending) with sentiment scores derived from customer reviews.

## 2. Requirements

* Python 3.x
* pandas
* scikit-learn
* matplotlib
* textblob

Install dependencies:

```
pip install pandas scikit-learn matplotlib textblob
```
## 3. Dataset

The dataset used in this project is synthetically generated within the code.

Features include:
- Age
- Income
- Spending Score
- Review
- Sentiment Score (derived using TextBlob)

## 4. Methodology

1. Generate synthetic customer data
2. Apply sentiment analysis using TextBlob
3. Combine numerical + sentiment features
4. Perform clustering using K-Means
5. Visualize clusters

## 5. How to Run

Run the Python script:

```
python main.py
```

Or open and run the Jupyter Notebook.

## 6. Output

* Cluster labels for each customer
* Scatter plot of customer segmentation
* Sentiment scores

## 7. Notes

* Random data is used for demonstration
* Results may vary slightly due to randomness
* Sentiment analysis is rule-based and has limitations
