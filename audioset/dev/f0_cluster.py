import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
# Replace 'your_file_path.csv' with the path to your CSV file
df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/dev/mean_f0_target_clean_normalised.csv')

# Assigning a unique color to each age group
age_group_colors = {
    "KITTEN": "blue",
    "YOUNG": "green",
    "ADULT": "red",
    "SENIOR": "purple"
}

# Adding a color column based on the 'Target' column
df['Color'] = df['Target'].map(age_group_colors)

# Creating the scatter plot
plt.figure(figsize=(10, 6))

# Plotting each age group separately to have different colors
for age_group, color in age_group_colors.items():
    subset = df[df['Target'] == age_group]
    plt.scatter(subset.index, subset['MeanF0'], s=100, c=color, label=age_group)

plt.title('Mean Frequency by Age Group')
plt.xlabel('Index')
plt.ylabel('Mean Frequency (Hz)')
plt.legend(title="Age Group")
plt.grid(True)

plt.show()

#
# from sklearn.cluster import KMeans
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the dataset
# # Replace 'your_file_path.csv' with the path to your CSV file
# df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/dev/mean_f0_target_clean.csv')
#
# # Selecting the feature for clustering (MeanF0 values)
# X = df[['MeanF0']].values
#
# # Performing K-Means clustering
# # You can adjust the number of clusters 'n_clusters' based on your dataset
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
#
# # Assigning cluster labels to each data point
# df['Cluster'] = kmeans.labels_
#
# # Plotting the clusters
# plt.figure(figsize=(10, 6))
#
# # Assigning a color to each cluster
# colors = ['blue', 'green', 'red', 'purple']
#
# for i in range(kmeans.n_clusters):
#     # Selecting data points that belong to the current cluster
#     subset = df[df['Cluster'] == i]
#     plt.scatter(subset.index, subset['MeanF0'], s=100, c=colors[i], label=f'Cluster {i}')
#
# plt.title('Mean Frequency Clustering')
# plt.xlabel('Index')
# plt.ylabel('Mean Frequency (Hz)')
# plt.legend(title="Cluster")
# plt.grid(True)
#
# plt.show()


# from sklearn.cluster import KMeans
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load your dataset
# df = pd.read_csv('/Users/astrid/PycharmProjects/tensorflow-fork/research/audioset/dev/mean_f0_target_clean.csv')  # Make sure to update this path with the actual location of your CSV file
#
# # Extracting the feature for clustering
# X = df[['MeanF0']].values  # Using 'MeanF0' as the feature for clustering
#
# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)  # Adjust 'n_clusters' as needed
#
# # Assign the cluster labels to the DataFrame
# df['Cluster'] = kmeans.labels_
#
# # Mapping each target class to a specific color
# target_colors = {
#     "KITTEN": "blue",
#     "YOUNG": "green",
#     "ADULT": "red",
#     "SENIOR": "purple"
# }
#
# # Adding a 'Color' column based on the 'Target' column for visualization
# df['Color'] = df['Target'].apply(lambda x: target_colors[x])
#
# # Plotting the results
# plt.figure(figsize=(10, 6))
#
# # Scatter plot of the data points colored by their original target class
# for target, color in target_colors.items():
#     subset = df[df['Target'] == target]
#     plt.scatter(subset.index, subset['MeanF0'], s=100, color=color, label=target)
#
# plt.title('MeanF0 by Target Class with K-Means Clusters')
# plt.xlabel('Index')
# plt.ylabel('MeanF0')
# plt.legend(title="Target Class")
# plt.grid(True)
#
# plt.show()


