from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Perform clustering on the embeddings
n_clusters = 4  # Choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
embeddings_df['cluster'] = kmeans.fit_predict(embeddings_df.drop(['cat_id', 'target'], axis=1))

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings_df.drop(['cluster', 'target', 'cat_id'], axis=1))
embeddings_df['PCA1'] = reduced_embeddings[:, 0]
embeddings_df['PCA2'] = reduced_embeddings[:, 1]

# Define target color palette
custom_palette = {'KITTEN': 'red', 'YOUNG': 'pink', 'SENIOR': 'blue', 'ADULT': 'purple'}

# Visualize the clusters with PCA components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=embeddings_df, palette='viridis', alpha=0.7)
plt.title('Cluster Visualization with PCA Components')
plt.show()

# Overlay the target classes for reference
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=embeddings_df, palette=custom_palette, alpha=0.7, legend='full')
plt.title('Target Class Visualization with PCA Components')
plt.show()