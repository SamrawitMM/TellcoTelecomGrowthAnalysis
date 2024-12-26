import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


from scripts.utility import read_csv_file, detect_outliers_iqr, analyze_handsets_data, find_high_correlation_pairs, calculate_decile, get_important_features
from scripts.plot import plot_histograms, plot_boxplots, plot_bar_chart, plot_pie_chart, plot_stacked_bar, plot_correlation_heatmap


# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file
data_path = os.path.join(script_dir, '../../data/preprocessed.csv')
data_path = os.path.abspath(data_path)  # Ensure it's an absolute path

# Use the data_path to load the file
telecom_data = read_csv_file(data_path)
telecom_data = telecom_data.get("data")

# Streamlit Page Title
st.title("User Engagement Analysis")

# Display Data
st.subheader("Raw Telecom Data")
st.write(telecom_data.head())

# Aggregate metrics by 'MSISDN/Number' (customer ID)
aggregated_data = telecom_data.groupby('MSISDN/Number').agg(
    session_frequency=('Bearer Id', 'count'),  # Count of sessions by customer
    total_session_duration=('Dur. (ms)', 'sum'),  # Total session duration (in ms)
    total_dl_traffic=('Total DL (Bytes)', 'sum'),  # Total data traffic from DL
    total_ul_traffic=('Total UL (Bytes)', 'sum')   # Total data traffic from UL
).reset_index()

# Add the total traffic by combining 'Total DL (Bytes)' and 'Total UL (Bytes)'
aggregated_data['total_traffic'] = aggregated_data['total_dl_traffic'] + aggregated_data['total_ul_traffic']

# Display the top 10 customers by each metric
top_10_session_frequency = aggregated_data.nlargest(10, 'session_frequency')
top_10_total_session_duration = aggregated_data.nlargest(10, 'total_session_duration')
top_10_total_traffic = aggregated_data.nlargest(10, 'total_traffic')

print("Top 10 customers by session frequency:")
print(top_10_session_frequency)

print("\nTop 10 customers by total session duration:")
print(top_10_total_session_duration)

print("\nTop 10 customers by total traffic:")
print(top_10_total_traffic)

top_10_session_frequency['MSISDN/Number'] = top_10_session_frequency['MSISDN/Number'].astype(int)


# Set up the visual style
sns.set(style="whitegrid")

# 1. Top 10 customers by session frequency
plot_bar_chart(data=top_10_session_frequency, x_col='MSISDN/Number', y_col='session_frequency', 
               x_label='Customer ID (MSISDN)', y_label='Session Frequency', 
               title='Top 10 Customers by Session Frequency', color_palette='Blues_d')

# 2. Top 10 customers by total session duration
plot_bar_chart(data=top_10_total_session_duration, x_col='MSISDN/Number', y_col='total_session_duration', 
               x_label='Customer ID (MSISDN)', y_label='Total Session Duration (ms)', 
               title='Top 10 Customers by Total Session Duration', color_palette='Greens_d')

# 3. Top 10 customers by total traffic
plot_bar_chart(data=top_10_total_traffic, x_col='MSISDN/Number', y_col='total_traffic', 
               x_label='Customer ID (MSISDN)', y_label='Total Traffic (Bytes)', 
               title='Top 10 Customers by Total Traffic (Download + Upload)', color_palette='Reds_d')



# Applications we are interested
applications = ['Google', 'YouTube', 'Games', 'Netflix', 'Email', 'Social Media', 'Other']

# Initialize an empty DataFrame to store results
aggregated_data = pd.DataFrame()

# Loop over each application and aggregate the data
for app in applications:
    if app == 'Google':
        app_data = telecom_data[['MSISDN/Number', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Start', 'Dur. (ms)']]
    elif app == 'YouTube':
        app_data = telecom_data[['MSISDN/Number', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Start', 'Dur. (ms)']]
    elif app == 'Games':
        app_data = telecom_data[['MSISDN/Number', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Start', 'Dur. (ms)']]

    # Aggregate data by MSISDN (customer ID)
    app_aggregated = app_data.groupby('MSISDN/Number').agg(
        session_frequency=('Start', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_upload_traffic=('Google UL (Bytes)' if app == 'Google' else 'Youtube UL (Bytes)' if app == 'YouTube' else 'Gaming UL (Bytes)', 'sum'),
        total_download_traffic=('Google DL (Bytes)' if app == 'Google' else 'Youtube DL (Bytes)' if app == 'YouTube' else 'Gaming DL (Bytes)', 'sum')
    ).reset_index()

    # Calculate the total traffic (upload + download)
    app_aggregated['total_traffic'] = app_aggregated['total_upload_traffic'] + app_aggregated['total_download_traffic']
    
    # Add the application name for tracking
    app_aggregated['App_Name'] = app
    
    # Append the aggregated data to the overall DataFrame
    aggregated_data = pd.concat([aggregated_data, app_aggregated])

# View aggregated data
print(aggregated_data.head())


# Aggregating the total traffic per application
app_engagement = aggregated_data.groupby(['App_Name', 'MSISDN/Number']).agg(
    total_traffic=('total_traffic', 'sum'),
    total_session_frequency=('session_frequency', 'sum'),
    total_session_duration=('total_session_duration', 'sum')
).reset_index()

# Sorting and displaying the top 10 most engaged users per application
top_10_google = app_engagement[app_engagement['App_Name'] == 'Google'].sort_values(by='total_traffic', ascending=False).head(10)
print("Top 10 most engaged Google users:")
print(top_10_google)

top_10_youtube = app_engagement[app_engagement['App_Name'] == 'YouTube'].sort_values(by='total_traffic', ascending=False).head(10)
print("Top 10 most engaged YouTube users:")
print(top_10_youtube)

top_10_games = app_engagement[app_engagement['App_Name'] == 'Games'].sort_values(by='total_traffic', ascending=False).head(10)
print("Top 10 most engaged Games users:")
print(top_10_games)


# Aggregating total traffic per application
app_usage = aggregated_data.groupby('App_Name').agg(
    total_traffic=('total_traffic', 'sum')
).reset_index()

# Sorting the applications by total traffic and selecting top 3
top_3_apps = app_usage.sort_values(by='total_traffic', ascending=False).head(3)

# Using the plot_bar_chart function to plot the bar chart
plot_bar_chart(data=top_3_apps, x_col='App_Name', y_col='total_traffic', 
               title='Top 3 Most Used Applications', x_label='Application', y_label='Total Traffic (Bytes)', 
               color_palette='viridis')
