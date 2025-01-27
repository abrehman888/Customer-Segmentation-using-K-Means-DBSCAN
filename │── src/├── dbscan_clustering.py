import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from data_preprocessing import load_and_preprocess_data

# Load dataset
df, scaled_features = load_and_preprocess_data("data/mall_customers.csv")

# Apply DBSCAN
db_scan = DBSCAN(eps=0.5, min_samples=4)
dbscan_labels = db_scan.fit_predict(scaled_features)

# Evaluate using silhouette score
try:
    dbscan_silhouette = silhouette_score(scaled_features, dbscan_labels)
    print(f'DBSCAN Silhouette Score: {dbscan_silhouette}')
except:
    print("Silhouette score not available (all points may be noise).")

# Save clustered dataset
df['DBSCAN_Cluster'] = dbscan_labels
df.to_csv("reports/dbscan_results.csv", index=False)
