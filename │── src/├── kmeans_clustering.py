import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_preprocessing import load_and_preprocess_data

# Load dataset
df, scaled_features = load_and_preprocess_data("data/mall_customers.csv")

# Find optimal clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Save elbow plot
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.savefig("reports/elbow_plot.png")
plt.show()

# Find optimal clusters using Silhouette Score
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f'Clusters: {i}, Silhouette Score: {score}')

# Apply K-Means with optimal clusters (e.g., k=5)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)
df['Cluster'] = kmeans_labels

# Save clustered dataset
df.to_csv("reports/kmeans_results.csv", index=False)
