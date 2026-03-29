import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from textblob import TextBlob

# Create dataset (add reviews)
data = {
    'Age': [22, 25, 47, 52, 46, 23, 30, 36, 40, 28],
    'Income': [15000, 29000, 48000, 52000, 50000, 16000, 30000, 35000, 42000, 27000],
    'Spending': [39, 81, 6, 77, 40, 76, 50, 60, 30, 70],
    'Review': [
        "I love this product", 
        "Amazing experience", 
        "Very bad quality", 
        "I am satisfied", 
        "Not worth the price",
        "Great service", 
        "Average product", 
        "Good but expensive", 
        "Terrible support", 
        "Pretty decent"
    ]
}

df = pd.DataFrame(data)

# ---------------- NLP PART ----------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment'] = df['Review'].apply(get_sentiment)

# ---------------- CLUSTERING ----------------
# Use Income, Spending, Sentiment
features = df[['Income', 'Spending', 'Sentiment']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# ---------------- PLOT ----------------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

# plot
ax.scatter(df['Income'], df['Spending'], c=df['Cluster'])

# labels
ax.set_xlabel('Income')
ax.set_ylabel('Spending Score')
ax.set_title('Customer Segmentation with Sentiment')

# FORCE GRID (rất mạnh)
ax.set_axisbelow(True)
ax.grid(True, which='both')

# Add minor grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5)

plt.show()