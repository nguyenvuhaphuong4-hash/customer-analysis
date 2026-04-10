import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from textblob import TextBlob

import random

# Sample reviews
positive_reviews = [
    "I love this product", "Amazing experience", "Great service",
    "Absolutely fantastic", "Highly recommend it", "Excellent quality",
    "Really happy with this", "Superb experience", "Loved it", "Very satisfied"
]

negative_reviews = [
    "Very bad quality", "Terrible support", "Not worth the price",
    "Worst purchase ever", "Very disappointing", "Bad service",
    "I regret buying this", "Not impressed", "Poor experience", "Awful product"
]

neutral_reviews = [
    "Average product", "It works fine", "Pretty decent",
    "Not bad at all", "Could be better", "Just okay",
    "Decent but pricey", "Nothing special", "Fair enough", "Okay experience"
]

data = {
    'Age': [],
    'Income': [],
    'Spending': [],
    'Review': []
}

for _ in range(100):
    age = random.randint(20, 60)
    income = random.randint(15000, 60000)
    spending = random.randint(1, 100)

    sentiment_type = random.choice(['pos', 'neg', 'neu'])

    if sentiment_type == 'pos':
        review = random.choice(positive_reviews)
    elif sentiment_type == 'neg':
        review = random.choice(negative_reviews)
    else:
        review = random.choice(neutral_reviews)

    data['Age'].append(age)
    data['Income'].append(income)
    data['Spending'].append(spending)
    data['Review'].append(review)

# Check sample
import pandas as pd
df = pd.DataFrame(data)
print(df.head())

df = pd.DataFrame(data)

# NLP PART
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment'] = df['Review'].apply(get_sentiment)

# CLUSTERING
# Use Income, Spending, Sentiment
features = df[['Income', 'Spending', 'Sentiment']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# PLOT

fig, ax = plt.subplots(figsize=(6, 5))

# plot
ax.scatter(df['Income'], df['Spending'], c=df['Cluster'])

# labels
ax.set_xlabel('Income')
ax.set_ylabel('Spending Score')
ax.set_title('Customer Segmentation with Sentiment')

# FORCE GRID
ax.set_axisbelow(True)
ax.grid(True, which='both')

# Add minor grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5)

plt.show()