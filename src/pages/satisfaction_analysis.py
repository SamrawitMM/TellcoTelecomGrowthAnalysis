import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from scripts.utility import read_csv_file, detect_outliers_iqr, analyze_handsets_data, find_high_correlation_pairs, calculate_decile, get_important_features
from scripts.plot import plot_histograms, plot_boxplots, plot_bar_chart, plot_pie_chart, plot_stacked_bar, plot_correlation_heatmap
# Load data
# data_path = './data/preprocessed.csv'

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file
data_path = os.path.join(script_dir, '../../data/preprocessed.csv')
data_path = os.path.abspath(data_path)  # Ensure it's an absolute path

# Use the data_path to load the file
telecom_data = read_csv_file(data_path)
telecom_data = telecom_data.get("data")

# Streamlit Page Title
st.title("Satisfaction Analysis")

# Preprocessing: Scaling the numerical features for clustering
features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
            'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
X = telecom_data[features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering to identify engagement and experience clusters
kmeans = KMeans(n_clusters=2, random_state=42)
telecom_data['cluster'] = kmeans.fit_predict(X_scaled)

# Assuming cluster 0 corresponds to low engagement and cluster 1 corresponds to worst experience
engagement_centroid = kmeans.cluster_centers_[0]  # cluster 0 centroid
experience_centroid = kmeans.cluster_centers_[1]  # cluster 1 centroid

# Calculate Euclidean distances for engagement and experience scores
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

telecom_data['engagement_score'] = [euclidean_distance(row, engagement_centroid) for row in X_scaled]
telecom_data['experience_score'] = [euclidean_distance(row, experience_centroid) for row in X_scaled]

# Print the first few rows of the result
telecom_data[['Bearer Id', 'engagement_score', 'experience_score']].head()
telecom_data['satisfaction_score'] = (telecom_data['engagement_score'] + telecom_data['experience_score']) / 2

# Combine engagement and experience scores for clustering
X_clustering = telecom_data[['engagement_score', 'experience_score']]

# Perform K-Means with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
telecom_data['engagement_experience_cluster'] = kmeans.fit_predict(X_clustering)

# Display the clustering results
telecom_data[['Bearer Id', 'engagement_score', 'experience_score', 'engagement_experience_cluster']].tail(20)


cluster_averages = telecom_data.groupby('engagement_experience_cluster')[['satisfaction_score', 'experience_score']].mean()


# ax = cluster_averages.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
# ax.set_title('Average Satisfaction and Experience Scores per Cluster')
# ax.set_xlabel('Engagement Experience Cluster')
# ax.set_ylabel('Scores')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()
# st.pyplot()


# import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
cluster_averages.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
ax.set_title('Average Satisfaction and Experience Scores per Cluster')
ax.set_xlabel('Engagement Experience Cluster')
ax.set_ylabel('Scores')
plt.xticks(rotation=0)
plt.tight_layout()

