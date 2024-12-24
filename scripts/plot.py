import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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



def plot_bar_chart(data, x_col, y_col, x_label="Number of Users", y_label="Handset Type", 
                   title="Top 10 Handsets", color_palette="viridis", horizontal=False):
    """
    Function to plot a customizable bar chart with an option for vertical or horizontal bars.
    
    Parameters:
    - data: The data to plot, which should be a pandas DataFrame or Series.
    - x_col: The column name to use for the X axis.
    - y_col: The column name to use for the Y axis.
    - x_label: The label for the X axis.
    - y_label: The label for the Y axis.
    - title: The title of the chart.
    - color_palette: The color palette to use for the bars.
    - horizontal: Boolean indicating whether the bars should be horizontal (True) or vertical (False).
    """
    # Check if the data is a DataFrame or Series and plot accordingly
    plt.figure(figsize=(12, 6))
    
    if isinstance(data, pd.DataFrame):
        # Plotting a DataFrame (expects two columns)
        if horizontal:
            # Horizontal bar plot
            sns.barplot(x=data[y_col], y=data[x_col].astype(str), palette=color_palette)
            plt.xlabel(y_label)
            plt.ylabel(x_label)
        else:
            # Vertical bar plot
            sns.barplot(x=data[x_col], y=data[y_col], palette=color_palette)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
    
    elif isinstance(data, pd.Series):
        # Plotting a Series (expects index for x-axis and values for y-axis)
        if horizontal:
            # Horizontal bar plot
            sns.barplot(x=data.values, y=data.index.astype(str), palette=color_palette)
            plt.xlabel(y_label)
            plt.ylabel(x_label)
        else:
            # Vertical bar plot
            sns.barplot(x=data.index.astype(str), y=data.values, palette=color_palette)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
    
    else:
        raise ValueError("Data must be a pandas DataFrame or Series.")
    
    # Set the plot labels and title
    plt.title(title)
    
    # Rotate x-axis labels if they are too long (for better readability)
    if not horizontal:
        plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability and align right
    
    plt.tight_layout()
    plt.show()



# Customizable function to plot a pie chart
def plot_pie_chart(data, title="Top 3 Handset Manufacturers", colors=None, startangle=140, autopct='%1.1f%%'):
    """
    Function to plot a customizable pie chart.
    
    Parameters:
    - data: The data to plot, which should be a series with values for the pie chart.
    - title: The title of the pie chart.
    - colors: The colors to use in the pie chart (optional).
    - startangle: The angle at which to start the pie chart.
    - autopct: The format for displaying percentage values.
    """
    if colors is None:
        colors = sns.color_palette("pastel")[:len(data)]
        
    plt.figure(figsize=(8, 8))
    plt.pie(
        data.values,
        labels=data.index,
        autopct=autopct,
        startangle=startangle,
        colors=colors
    )
    plt.title(title)
    plt.show()



# Stacked bar plot
def plot_stacked_bar(data, x_col, y_cols, labels=None, colors=None, title=None, xlabel=None, ylabel=None, figsize=(10, 6), rotation=90, ha='right', va='center'):
    """
    Creates a stacked bar plot using seaborn for the given DataFrame.

    Parameters:
    - data (DataFrame): The data containing the columns to plot.
    - x_col (str): Column name for the x-axis (categorical variable).
    - y_cols (list): List of column names for the y-axis (values to stack).
    - labels (list): Labels for the legend. Defaults to the y_cols names.
    - colors (list): Colors for the stacked bars. Defaults to seaborn's default colors.
    - title (str): Title of the plot. Defaults to None.
    - xlabel (str): Label for the x-axis. Defaults to None.
    - ylabel (str): Label for the y-axis. Defaults to None.
    - figsize (tuple): Figure size. Defaults to (10, 6).
    - rotation (int): Rotation angle for x-axis labels. Defaults to 90.
    - ha (str): Horizontal alignment of x-axis labels. Defaults to 'right'.
    - va (str): Vertical alignment of x-axis labels. Defaults to 'center'.

    Returns:
    - None
    """
    # Ensure y_cols exist in the DataFrame
    for col in y_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the data.")

    # Default labels and colors
    if labels is None:
        labels = y_cols
    if colors is None:
        colors = sns.color_palette("Set2", len(y_cols))

    # Create the stacked bar plot
    plt.figure(figsize=figsize)

    # Initialize the bottom for stacking
    bottom = [0] * len(data)

    # Plot each y_col on top of the previous one using the bottom argument for stacking
    for idx, y_col in enumerate(y_cols):
        sns.barplot(x=data[x_col], y=data[y_col], color=colors[idx], label=labels[idx], bottom=bottom)
        bottom = [b + data[y_col][i] for i, b in enumerate(bottom)]  # Update bottom for stacking

    # Customize the plot
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Adjust x-axis labels
    plt.xticks(rotation=rotation, ha=ha, va=va)
    plt.gca().tick_params(axis='x', pad=15)  # Increase padding for better spacing

    # Customize legend
    plt.legend(title="Data Type")
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# Plot a stacked bar chart for the given data
def plot_stacked_bar(data, title, x_col, y_cols, labels, colors=None, figsize=(10, 6), 
                     xlabel=None, ylabel=None, rotation=45, ha='center', va='center'):
    """
    Function to plot a stacked bar chart for the given data.
    
    Parameters:
    - data: The DataFrame containing the data to plot
    - title: Title of the plot
    - x_col: The column to use for the x-axis (e.g., 'IMSI')
    - y_cols: List of columns to stack on the y-axis
    - labels: List of labels corresponding to the y_cols
    - colors: Optional list of colors to use for the bars (default is None)
    - figsize: Optional tuple specifying the figure size (default is (10, 6))
    - xlabel: Optional label for the x-axis (default is None)
    - ylabel: Optional label for the y-axis (default is None)
    - rotation: Optional rotation angle for x-axis labels (default is 0)
    - ha: Horizontal alignment for legend labels (default is 'center')
    - va: Vertical alignment for legend labels (default is 'center')
    """
    # Plotting the stacked bar chart
    ax = data.set_index(x_col)[y_cols].plot(kind='bar', stacked=True, figsize=figsize, cmap='viridis', color=colors)

    # Customize the plot
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticklabels(data[x_col], rotation=rotation)  # Rotate x-axis labels for better visibility

    # Add custom labels if provided
    if labels:
        ax.legend(labels, title='Applications', loc='upper right', fontsize='small',
                  bbox_to_anchor=(1.0, 1.0), frameon=True, 
                  handlelength=2.0, labelspacing=1.5)

        # Adjust label alignment inside the legend box using ha='right' and va='top'
        for label in ax.get_legend().get_texts():
            label.set_horizontalalignment(ha)  # Align horizontally as per parameter
            label.set_verticalalignment(va)  # Align vertically as per parameter

    # Adjust layout to prevent label and legend overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust space on the right for the legend

    # Show the plot
    plt.show()

# Plot a correlation heatmap for selected numeric columns
def plot_correlation_heatmap(data, columns, figsize=(12, 8), cmap='coolwarm'):
    """
    Function to plot a correlation heatmap for selected numeric columns.

    Parameters:
    - data: DataFrame containing the data to plot
    - columns: List of column names to include in the correlation matrix
    - figsize: Tuple specifying the figure size (default is (12, 8))
    - cmap: Colormap for the heatmap (default is 'coolwarm')
    """
    # Selecting relevant numeric columns for correlation
    correlation_data = data[columns]
    
    # Calculating the correlation matrix
    correlation_matrix = correlation_data.corr()

    # Creating the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, cbar=True)
    plt.title('Correlation Heatmap')
    plt.show()
