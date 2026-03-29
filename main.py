import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create dataset
data = {
    'Age': [22, 25, 47, 52, 46, 23, 30, 36, 40, 28],
    'Income': [15000, 29000, 48000, 52000, 50000, 16000, 30000, 35000, 42000, 27000],
    'Spending': [39, 81, 6, 77, 40, 76, 50, 60, 30, 70]
}

df = pd.DataFrame(data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Create figure (important)
plt.figure(figsize=(6, 5))

# Plot data points
plt.scatter(df['Income'], df['Spending'], c=df['Cluster'])

# Labels and title
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')

# Add visible grid
plt.grid(True, linestyle='--', linewidth=1, color='gray')

# Show plot
plt.show()