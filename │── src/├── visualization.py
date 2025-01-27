import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from data_preprocessing import load_and_preprocess_data
from kmeans_clustering import kmeans_labels
from dbscan_clustering import dbscan_labels

# Load dataset
df, _ = load_and_preprocess_data("data/mall_customers.csv")

# 2D K-Means Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Annual Income (k$)"], 
                y=df["Spending Score (1-100)"], 
                hue=kmeans_labels, 
                palette='viridis', 
                s=100)
plt.title("K-Means Clustering (2D)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.savefig("reports/kmeans_2D.png")
plt.show()

# 2D DBSCAN Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Annual Income (k$)"], 
                y=df["Spending Score (1-100)"], 
                hue=dbscan_labels, 
                palette='plasma', 
                s=100)
plt.title("DBSCAN Clustering (2D)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.savefig("reports/dbscan_2D.png")
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["Annual Income (k$)"], 
           df["Spending Score (1-100)"], 
           df.index, 
           c=kmeans_labels, cmap='viridis', s=50)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_zlabel("Customer Index")
ax.set_title("3D Visualization of K-Means Clustering")
plt.savefig("reports/kmeans_3D.png")
plt.show()
