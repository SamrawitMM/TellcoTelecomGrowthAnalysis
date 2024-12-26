import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# print(os.getcwd())

# sys.path.append(os.path.abspath('../scripts'))
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
st.title("Experience Analysis")



# Aggregate per customer
aggregated_data = telecom_data.groupby("MSISDN/Number").agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean',
    'Handset Type': lambda x: x.mode()[0] if not x.mode().empty else np.nan
}).reset_index()

aggregated_data.rename(columns={
    'TCP DL Retrans. Vol (Bytes)': 'Avg TCP Retransmission',
    'Avg RTT DL (ms)': 'Avg RTT',
    'Avg Bearer TP DL (kbps)': 'Avg Throughput'
}, inplace=True)

# Histogram for Throughput Distribution
plt.figure(figsize=(12, 6))
throughput_values = telecom_data['Avg Bearer TP DL (kbps)'].dropna()
plt.hist(throughput_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Throughput Distribution (Avg Bearer TP DL in kbps)')
plt.xlabel('Average Throughput (kbps)')
plt.ylabel('Frequency')
plt.show()
st.pyplot()


# Histogram for TCP Retransmission Distribution
plt.figure(figsize=(12, 6))
tcp_retrans_values = telecom_data['TCP DL Retrans. Vol (Bytes)'].dropna()
plt.hist(tcp_retrans_values, bins=20, color='orange', edgecolor='black', alpha=0.7)
plt.title('TCP Retransmission Distribution (TCP DL Retrans. Vol in Bytes)')
plt.xlabel('TCP Retransmission (Bytes)')
plt.ylabel('Frequency')
plt.show()
st.pyplot()



# Select numerical features for clustering
clustering_data = aggregated_data[['Avg TCP Retransmission', 'TCP UL Retrans. Vol (Bytes)', 
                                   'Avg RTT', 'Avg RTT UL (ms)', 'Avg Throughput', 'Avg Bearer TP UL (kbps)']]

# Apply Min-Max scaling (scaling values between 0 and 1)
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Perform K-Means clustering (let's choose 3 clusters for now)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)

# Add the cluster labels to the original dataset
aggregated_data['Cluster'] = clusters

# Visualize the clusters using a pairplot
sns.pairplot(aggregated_data, hue='Cluster', vars=['Avg TCP Retransmission', 'Avg RTT', 'Avg Throughput'])
plt.show()
st.pyplot()


# Print the cluster centers (mean values of each cluster)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=clustering_data.columns)
print("Cluster Centers (mean values of each cluster):")
print(cluster_centers)

# Summary of each cluster: Filter only numeric columns
numeric_cols = aggregated_data.select_dtypes(include=['number']).columns
cluster_means = aggregated_data[numeric_cols].groupby(aggregated_data['Cluster']).mean()

# Print the cluster summary
print("\nCluster Summary:")
print(cluster_means)

