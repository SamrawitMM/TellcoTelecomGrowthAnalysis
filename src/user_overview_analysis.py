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
data_path = os.path.join(script_dir, '../data/preprocessed.csv')
data_path = os.path.abspath(data_path)  # Ensure it's an absolute path

# Use the data_path to load the file
telecom_data = read_csv_file(data_path)
telecom_data = telecom_data.get("data")

# Streamlit Page Title
st.title("User Overview Analysis")

# Display Data
st.subheader("Raw Telecom Data")
st.write(telecom_data.head())



# Handset Data Analysis
top_10_handsets, top_3_manufacturers, top_5_handsets_per_manufacturer = analyze_handsets_data(telecom_data)

# Display Top 10 Handsets
st.subheader("Top 10 Handsets Used by Customers")
plot_bar_chart(
    data=top_10_handsets, 
    x_col='Handset_Type', 
    y_col='Number_of_Users',
    title="Top 10 Handsets Used by Customers", 
    x_label="Handset Type", 
    y_label="Number of Users", 
    color_palette="viridis", 
    horizontal=True
)

# Display Pie Chart for Top 3 Manufacturers
st.subheader("Top 3 Handset Manufacturers")
plot_pie_chart(top_3_manufacturers, title="Top 3 Handset Manufacturers", colors=sns.color_palette("pastel")[:3])

# Loop to plot top 5 handsets for each manufacturer
for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
    st.subheader(f"Top 5 Handsets for Manufacturer: {manufacturer}")
    plot_bar_chart(
        data=handsets,  
        x_col='Handset_Type', 
        y_col='Number_of_Users',
        title=f"Top 5 Handsets for Manufacturer: {manufacturer}", 
        x_label="Handset Type", 
        y_label="Number of Users", 
        color_palette="coolwarm", 
        horizontal=True
    )

# Aggregated Data
telecom_data['Total Data Volume (Bytes)'] = telecom_data['Total DL (Bytes)'] + telecom_data['Total UL (Bytes)']
aggregated_data = telecom_data.groupby('IMSI').agg(
    Number_of_xDR_Sessions=('Bearer Id', 'nunique'),
    Total_Session_Duration=('Dur. (ms)', 'sum'),
    Total_Download_Data=('Total DL (Bytes)', 'sum'),
    Total_Upload_Data=('Total UL (Bytes)', 'sum'),
    Total_Data_Volume=('Total Data Volume (Bytes)', 'sum')
).reset_index()

# Top 10 Users by xDR Sessions
top_xdr_users = aggregated_data.nlargest(10, 'Number_of_xDR_Sessions')
st.subheader("Top 10 Users by Number of xDR Sessions")
plot_bar_chart(
    data=top_xdr_users,
    x_col='IMSI',
    y_col='Number_of_xDR_Sessions',
    x_label="User (IMSI)",
    y_label="Number of xDR Sessions",
    title="Top 10 Users by Number of xDR Sessions",
    color_palette="coolwarm",
    horizontal=False
)

# Top 10 Users by Total Data Volume
top_data_users = aggregated_data.nlargest(10, 'Total_Data_Volume')
st.subheader("Top 10 Users by Total Data Volume")
plot_bar_chart(
    data=top_data_users,
    x_col='IMSI',
    y_col='Total_Data_Volume',
    x_label="User (IMSI)",
    y_label="Total Data Volume",
    title="Top 10 Users by Total Data Volume",
    color_palette="coolwarm",
    horizontal=False
)

# Add total data volume column
telecom_data['Total Data Volume (Bytes)'] = telecom_data['Total DL (Bytes)'] + telecom_data['Total UL (Bytes)']

# Aggregating per user
aggregated_data = telecom_data.groupby('IMSI').agg(
    Number_of_xDR_Sessions=('Bearer Id', 'nunique'),
    Total_Session_Duration=('Dur. (ms)', 'sum'),
    Total_Download_Data=('Total DL (Bytes)', 'sum'),
    Total_Upload_Data=('Total UL (Bytes)', 'sum'),
    Total_Data_Volume=('Total Data Volume (Bytes)', 'sum'),
    Social_Media_DL=('Social Media DL (Bytes)', 'sum'),
    Social_Media_UL=('Social Media UL (Bytes)', 'sum'),
    Google_DL=('Google DL (Bytes)', 'sum'),
    Google_UL=('Google UL (Bytes)', 'sum'),
    Email_DL=('Email DL (Bytes)', 'sum'),
    Email_UL=('Email UL (Bytes)', 'sum'),
    Youtube_DL=('Youtube DL (Bytes)', 'sum'),
    Youtube_UL=('Youtube UL (Bytes)', 'sum'),
    Netflix_DL=('Netflix DL (Bytes)', 'sum'),
    Netflix_UL=('Netflix UL (Bytes)', 'sum'),
    Gaming_DL=('Gaming DL (Bytes)', 'sum'),
    Gaming_UL=('Gaming UL (Bytes)', 'sum'),
    Other_DL=('Other DL (Bytes)', 'sum'),
    Other_UL=('Other UL (Bytes)', 'sum')
).reset_index()

# Display the aggregated data
aggregated_data.head(10)

# Top 10 users by total data volume
top_data_users = aggregated_data.nlargest(10, 'Total_Data_Volume')



st.subheader("Top 10 Applications based on data usage")

application_data = aggregated_data[['Social_Media_DL', 'Google_DL', 'Email_DL', 'Youtube_DL', 'Netflix_DL', 'Gaming_DL', 'Other_DL']].sum()

plot_bar_chart(
    data=application_data,
    x_col='IMSI',  # IMSI will be on the x-axis
    y_col='Total_Data_Volume',  # Number_of_xDR_Sessions will be on the y-axis
    x_label="Applications",  # Label for the x-axis
    y_label="Total Da|ta Usage (Bytes)",  # Label for the y-axis
    title="Top 10 Applications based on data usage",  # Title of the chart
    color_palette="coolwarm",  # Color palette for the bars
    horizontal=False  # Use vertical bars
)



# Aggregating Total Download Data and Total Upload Data by IMSI (User)
download_upload_data = aggregated_data[['IMSI', 'Total_Download_Data', 'Total_Upload_Data']]

# Display the aggregated data for all users
# print("Total Download and Upload Data by User:")
download_upload_data[['IMSI', 'Total_Download_Data', 'Total_Upload_Data']]

# Sort the data based on Total Download Data and Total Upload Data in descending order
download_upload_data_sorted = download_upload_data.sort_values(by=['Total_Download_Data', 'Total_Upload_Data'], ascending=False)

# Select the top 10 users
top_10_users = download_upload_data_sorted.head(10)

# Reset index to avoid potential issues with x-axis plotting
top_10_users = top_10_users.reset_index(drop=True)  # Reset the index here

st.subheader("Top 10 Users: Total Download and Upload Data")

# Use the function to plot the stacked bar plot
plot_stacked_bar(
    data=top_10_users,
    x_col='IMSI',
    y_cols=['Total_Download_Data', 'Total_Upload_Data'],
    labels=['Download Data', 'Upload Data'],
    colors=['blue', 'orange'],
    title='Top 10 Users: Total Download and Upload Data',
    xlabel='IMSI',
    ylabel='Data (Bytes)',
    figsize=(10, 6),
    rotation=45,  # Rotate labels 45 degrees
    ha='right',  # Right-align the labels
    va='top'  # Align the labels at the top for better readability
)



# data preparation 
telecom_data['Total_Social_Media_Data'] = telecom_data['Social Media DL (Bytes)'] + telecom_data['Social Media UL (Bytes)']
telecom_data['Total_Google_Data'] = telecom_data['Google DL (Bytes)'] + telecom_data['Google UL (Bytes)']
telecom_data['Total_Email_Data'] = telecom_data['Email DL (Bytes)'] + telecom_data['Email UL (Bytes)']
telecom_data['Total_Youtube_Data'] = telecom_data['Youtube DL (Bytes)'] + telecom_data['Youtube UL (Bytes)']
telecom_data['Total_Netflix_Data'] = telecom_data['Netflix DL (Bytes)'] + telecom_data['Netflix UL (Bytes)']
telecom_data['Total_Gaming_Data'] = telecom_data['Gaming DL (Bytes)'] + telecom_data['Gaming UL (Bytes)']
telecom_data['Total_Other_Data'] = telecom_data['Other DL (Bytes)'] + telecom_data['Other UL (Bytes)']

# Select the columns for plotting: User and Total Data Volume for each application
data_volume_by_app = telecom_data[['IMSI', 
                                   'Total_Social_Media_Data', 
                                   'Total_Google_Data', 
                                   'Total_Email_Data', 
                                   'Total_Youtube_Data', 
                                   'Total_Netflix_Data', 
                                   'Total_Gaming_Data', 
                                   'Total_Other_Data']]

# Sort the data by the total data volume across all applications for each user (in descending order)
data_volume_by_app_sorted = data_volume_by_app.set_index('IMSI').sum(axis=1).sort_values(ascending=False).head(10)

# Get the top 10 users
top_10_users = data_volume_by_app.loc[data_volume_by_app['IMSI'].isin(data_volume_by_app_sorted.index)]

st.subheader("Top 10 Users: Data Volumne Breakdown by Application")

# Now use the plot_stacked_bar function with the correct parameters
plot_stacked_bar(
    data=top_10_users,
    title='Top 10 Users: Data Volume Breakdown by Application',
    x_col='IMSI',
    y_cols=[
        'Total_Social_Media_Data',
        'Total_Google_Data',
        'Total_Email_Data',
        'Total_Youtube_Data',
        'Total_Netflix_Data',
        'Total_Gaming_Data',
        'Total_Other_Data'
    ],
    labels=[
        'Social Media Data',
        'Google Data',
        'Email Data',
        'YouTube Data',
        'Netflix Data',
        'Gaming Data',
        'Other Data'
    ],
    figsize=(12, 6)
)

