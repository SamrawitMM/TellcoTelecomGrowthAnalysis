import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def plot_histograms(data, numerical_columns, cols=4, figsize=(30, 60)):
    """
    Function to plot histograms for numerical columns in a DataFrame.

    Parameters:
    - data: The DataFrame containing the data to be plotted.
    - numerical_columns: List of numerical columns to plot.
    - cols: Number of columns per row for the subplot grid (default is 4).
    - figsize: The figure size for the entire plot (default is (30, 60)).
    """
    # Calculate the number of rows needed based on the number of numerical columns
    rows = (len(numerical_columns) // cols) + (len(numerical_columns) % cols > 0)

    # Create subplots with adjusted figsize to make the plots larger
    fig, axes = plt.subplots(rows, cols, figsize=figsize)  # Adjusted figsize

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the numerical columns and plot each histogram
    for i, col in enumerate(numerical_columns):
        axes[i].hist(data[col], bins=50, edgecolor='black')
        axes[i].set_title(col, fontsize=14)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots if the number of columns is not a perfect multiple of 4
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Set the suptitle for the entire figure
    plt.suptitle('Histograms for Numerical Columns', fontsize=24)

    # Adjust the layout to make sure everything fits
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()
    st.pyplot(fig)



def plot_boxplots(data, numerical_columns, cols=4, figsize=(30, 60)):
    """
    Function to plot boxplots for numerical columns in a DataFrame.

    Parameters:
    - data: The DataFrame containing the data to be plotted.
    - numerical_columns: List of numerical columns to plot.
    - cols: Number of columns per row for the subplot grid (default is 4).
    - figsize: The figure size for the entire plot (default is (30, 60)).
    """
    # Calculate the number of rows needed based on the number of numerical columns
    rows = (len(numerical_columns) // cols) + (len(numerical_columns) % cols > 0)

    # Create subplots with adjusted figsize to make the plots larger
    fig, axes = plt.subplots(rows, cols, figsize=figsize)  # Adjusted figsize

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the numerical columns and plot each box plot
    for i, col in enumerate(numerical_columns):
        axes[i].boxplot(data[col].dropna(), vert=False)  # Drop NA values for plotting
        axes[i].set_title(col, fontsize=14)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots if the number of columns is not a perfect multiple of 4
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Set the suptitle for the entire figure
    plt.suptitle('Box Plots for Numerical Columns', fontsize=24)

    # Adjust the layout to make sure everything fits
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()
    st.pyplot(fig)



    # Function to perform analysis on handsets data
def analyze_handsets_data(telecom_data):
    """
    Function to analyze handsets data and return:
    - Top 10 handsets
    - Top 3 manufacturers
    - Top 5 handsets per manufacturer
    """
    # Identify the top 10 handsets used by customers
    top_10_handsets = telecom_data['Handset Type'].value_counts().head(10)

    # Identify the top 3 handset manufacturers
    top_3_manufacturers = telecom_data['Handset Manufacturer'].value_counts().head(3)

    # Identify the top 5 handsets per top 3 handset manufacturer
    top_5_handsets_per_manufacturer = {}
    for manufacturer in top_3_manufacturers.index:
        top_handsets = telecom_data[telecom_data['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = top_handsets

    return top_10_handsets, top_3_manufacturers, top_5_handsets_per_manufacturer


# Function to plot a customizable bar chart
def plot_bar_chart(data, x_col, y_col, x_label="Number of Users", y_label="Handset Type", 
                   title="Top 10 Handsets", color_palette="viridis", horizontal=False):
    """
    Function to plot a customizable bar chart with an option for vertical or horizontal bars.
    """
    # Check if the data is a DataFrame or Series and plot accordingly
    fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axis
    if isinstance(data, pd.DataFrame):
        if horizontal:
            sns.barplot(x=data[y_col], y=data[x_col].astype(str), palette=color_palette, ax=ax)
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        else:
            sns.barplot(x=data[x_col], y=data[y_col], palette=color_palette, ax=ax)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    
    elif isinstance(data, pd.Series):
        if horizontal:
            sns.barplot(x=data.values, y=data.index.astype(str), palette=color_palette, ax=ax)
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        else:
            sns.barplot(x=data.index.astype(str), y=data.values, palette=color_palette, ax=ax)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    
    else:
        raise ValueError("Data must be a pandas DataFrame or Series.")
    
    ax.set_title(title)
    if not horizontal:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate labels for better readability
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)


# Customizable function to plot a pie chart
def plot_pie_chart(data, title="Top 3 Handset Manufacturers", colors=None, startangle=140, autopct='%1.1f%%'):
    """
    Function to plot a customizable pie chart.
    """
    if colors is None:
        colors = sns.color_palette("pastel")[:len(data)]
        
    fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axis
    ax.pie(
        data.values,
        labels=data.index,
        autopct=autopct,
        startangle=startangle,
        colors=colors
    )
    ax.set_title(title)
    plt.show()
    st.pyplot(fig)


# Stacked bar plot with ax
def plot_stacked_bar(data, title, x_col, y_cols, labels, colors=None, figsize=(10, 6), 
                             xlabel=None, ylabel=None, rotation=45, ha='center', va='center'):
    """
    Function to plot a stacked bar chart with Axes object (ax).
    """
    fig, ax = plt.subplots(figsize=figsize)  # Create figure and axis
    data.set_index(x_col)[y_cols].plot(kind='bar', stacked=True, figsize=figsize, cmap='viridis', color=colors, ax=ax)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xticklabels(data[x_col], rotation=rotation)
    
    if labels:
        ax.legend(labels, title='Applications', loc='upper right', fontsize='small',
                  bbox_to_anchor=(1.0, 1.0), frameon=True, 
                  handlelength=2.0, labelspacing=1.5)
        for label in ax.get_legend().get_texts():
            label.set_horizontalalignment(ha)
            label.set_verticalalignment(va)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust space on the right for the legend
    plt.show()
    st.pyplot(fig)


# Plot a correlation heatmap for selected numeric columns
def plot_correlation_heatmap(data, columns, figsize=(12, 8), cmap='coolwarm'):
    """
    Function to plot a correlation heatmap for selected numeric columns.
    """
    correlation_data = data[columns]
    correlation_matrix = correlation_data.corr()

    fig, ax = plt.subplots(figsize=figsize)  # Create figure and axis
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, cbar=True, ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.show()
    st.pyplot(fig)


# Elbow curve plot for k-values and WCSS
def plot_elbow_curve(k_values, wcss):
    """
    Plots the Elbow Curve to visualize the optimal number of clusters.
    """
    fig, ax = plt.subplots(figsize=(8, 5))  # Create figure and axis
    ax.plot(k_values, wcss, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method for Optimal k')
    plt.show()
    st.pyplot(fig)


# Plot user engagement clusters with PCA
def plot_user_engagement_clusters(pca_features, cluster_labels, optimal_k):
    """
    Plots the user engagement clusters in the PCA space.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
    for cluster in range(optimal_k):
        cluster_points = pca_features[cluster_labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('User Engagement Clusters')
    ax.legend()
    plt.show()
    st.pyplot(fig)



