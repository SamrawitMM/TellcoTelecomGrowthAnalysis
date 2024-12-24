import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_csv_file(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Remove any 'Unnamed:' columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Get additional info
    column_names = data.columns.tolist()  
    row_count = data.shape[0]  
    
    return {
        'data': data,
        'column_names': column_names,
        'row_count': row_count
    }


def detect_outliers_iqr(data, skewed_cols, id_col):
    """
    Detect outliers in a DataFrame using the IQR method and return the list of unique IDs for outliers.

    Parameters:
    - data: DataFrame
    - skewed_cols: List of numerical columns to check for outliers.
    - id_col: The column name used as the identifier (e.g., IMSI).

    Returns:
    - outlier_ids: A list of unique IDs for rows identified as outliers.
    """
    outlier_ids = set()  # Use a set to store unique IDs

    for col in skewed_cols:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = 0
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_ids.update(outliers[id_col].unique())  # Add unique IDs of outliers
            
            print(f'Column: {col} - Number of outliers: {outliers.shape[0]}')
            print(f'Lower bound: {lower_bound} & Upper bound: {upper_bound}\n')
        else:
            print(f'Column {col} not found in the DataFrame.')

    return list(outlier_ids)  # Convert the set to a list


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




def find_high_correlation_pairs(data, threshold=0.80):
    """
    Function to find pairs of variables with a correlation above the specified threshold.

    Parameters:
    - data: DataFrame containing the data to analyze
    - threshold: Correlation threshold to filter pairs (default is 0.80)

    Returns:
    - DataFrame of variable pairs with correlation above the threshold
    """
    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Extract pairs of variables and their correlation
    correlation_pairs = (
        correlation_matrix
        .where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))  # Keep upper triangle and convert to boolean
        .stack()  # Convert to long format
        .reset_index()  # Reset index
    )

    # Rename columns for clarity
    correlation_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # Filter pairs with correlation above the threshold
    high_correlation_pairs = correlation_pairs[correlation_pairs['Correlation'] > threshold]

    return high_correlation_pairs


def find_high_correlation_pairs(data, threshold=0.80):
    """
    Function to find pairs of variables with a correlation above the specified threshold.

    Parameters:
    - data: DataFrame containing the data to analyze
    - threshold: Correlation threshold to filter pairs (default is 0.80)

    Returns:
    - DataFrame of variable pairs with correlation above the threshold
    """
    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Extract pairs of variables and their correlation
    correlation_pairs = (
        correlation_matrix
        .where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))  # Keep upper triangle and convert to boolean
        .stack()  # Convert to long format
        .reset_index()  # Reset index
    )

    # Rename columns for clarity
    correlation_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # Filter pairs with correlation above the threshold
    high_correlation_pairs = correlation_pairs[correlation_pairs['Correlation'] > threshold]

    return high_correlation_pairs


def calculate_decile(telecom_data):
    """
    Function to calculate total session duration per user, classify users into duration deciles,
    and compute total download and upload data per decile.

    Parameters:
    - telecom_data: DataFrame containing telecom data with columns 'IMSI', 'Dur. (ms)', 'Total DL (Bytes)', and 'Total UL (Bytes)'

    Returns:
    - DataFrame with total download + upload data per duration decile
    """
    # Calculate total session duration per user
    telecom_data['Total Duration'] = telecom_data.groupby('IMSI')['Dur. (ms)'].transform('sum')

    # Calculate deciles based on total duration
    deciles = pd.qcut(telecom_data['Total Duration'], 10, labels=False)  # 10 equal-sized bins

    # Add decile classification to the DataFrame
    telecom_data['Duration Decile'] = deciles

    # Compute the total data (DL + UL) per decile
    telecom_data['Total DL + UL'] = telecom_data['Total DL (Bytes)'] + telecom_data['Total UL (Bytes)']
    total_data_per_decile = telecom_data.groupby('Duration Decile')['Total DL + UL'].sum().reset_index()

    return total_data_per_decile



def get_important_features(df, n_top_features=10, handle_missing='mean'):
    """
    Function to perform PCA on a dataset and return the most important features based on PCA loadings.

    Parameters:
    - df: DataFrame containing the dataset to perform PCA on. 
    - n_top_features: Number of top features to return based on PCA loadings (default is 10).
    - handle_missing: Method to handle missing values in the data ('mean' for mean imputation, 'drop' to drop rows with missing values).

    Returns:
    - A DataFrame containing the most important features and their respective importance.
    """
    # Select numeric columns for PCA (select only features with float data type)
    df_selected = df.select_dtypes(include=['float64'])

    # Handle missing values (impute with mean or drop rows)
    if handle_missing == 'mean':
        df_selected = df_selected.fillna(df_selected.mean())
    elif handle_missing == 'drop':
        df_selected = df_selected.dropna()

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_selected)

    # Apply PCA to the dataset
    pca = PCA()
    pca.fit(df_scaled)

    # Get the PCA loadings (weights)
    loadings = pd.DataFrame(pca.components_, columns=df_selected.columns)

    # Calculate the absolute value of the loadings
    absolute_loadings = loadings.abs()

    # Sum the absolute loadings for each feature across all principal components
    feature_importance = absolute_loadings.sum(axis=0)

    # Create a DataFrame to view the importance of each feature
    feature_importance_df = pd.DataFrame({'Feature': df_selected.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Select the top N most important features
    top_features = feature_importance_df.head(n_top_features)

    return top_features