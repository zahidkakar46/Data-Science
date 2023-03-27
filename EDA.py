from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
def DESC():
    """
        This module that contains several functions for data manipulation
        and analysis using scikit-learn, scipy, pandas, numpy, matplotlib, and geopandas libraries.
        The module contains 15 functions that can be used to perform different data preprocessing
        and analysis tasks.

        **Module Set Characteristics:**

        :Number of Functions: 15

    Functions:
    ----------
        - datasource: Loads a csv file as a pandas DataFrame and returns it.
        - shapdatasource: Loads a csv file as a shap DataFrame and returns it.
        - remove_columns: Removes a list of columns from a pandas DataFrame.
        - change_datatype: Changes the datatype of specified columns in a pandas DataFrame.
        - check_for_missing: Checks if any missing values exist in a pandas DataFrame
          and returns a boolean value.
        - replace_null_with_zero: Replaces missing values in specified columns of a pandas DataFrame with zeros.
        - show_null_rows: Returns rows from a pandas DataFrame that contain missing values.
        - fill_nulls: Fills missing values in specified columns of a pandas DataFrame using a specified method.
        - count_rows: Returns the number of rows in a pandas DataFrame.
        - replace_values: Replaces specified values in specified columns of a pandas DataFrame with a new value.
        - outliers_count: Returns the number of outliers in a pandas DataFrame for a specified column.
        - outliers_details: Returns details of the outliers in a pandas DataFrame for a specified column.
        - plot_outliers: Generates a box plot to visualize the distribution of
          a column in a pandas DataFrame and highlight the outliers.
        - plot_outliers_bycategory: Generates a box plot to visualize the distribution of
          a column in a pandas DataFrame grouped by a specified categorical column and highlight the outliers.
        - plot_histo: Generates a histogram to visualize the distribution of a column in a pandas DataFrame.

    """

def load_dataset():
    """
    Loads the NRAP dataset from a CSV file and returns it as a pandas DataFrame.

    Returns:
        pandas.DataFrame: The NRAP dataset as a pandas DataFrame.
        
    **Data Set Characteristics:**

    :Number of Instances: 770 (every row is a contract)

    :Number of Attributes: 24

    :Attribute Information:
        - Region                 -string data type, contains regional location of the contract
        - Province               -string data type, contains provincial location of the contract
        - District               -string data type, contains district location of the contract
        - ContractID             -int data type, contains the id of contract, which a unique identifier 
        - Contractor             -string data type, contains the name of contractor
        - Road(KM)               -float data type, contains the kilometer of road of the contract
        - Bridge(M)              -float data type, contains the meter of bridge of the contract
        - Structure(M)           -float data type, contains the meter of small structure of the contract
        - Cost(USD)              -float data type, contains the $cost of the contract
        - ContractSignDate       -date data type, contains the sign date of the contract
        - PlanStartDate          -date data type, contains the plan start date of the contract
        - ActualStartDate        -date data type, contains the actual start date of the contract
        - PlanCompletionDate     -date data type, contains the plan completion date of the contract
        - ActualCompletionDate   -date data type, contains the actual completion date of the contract
        - Status                 -string data type, contains the status of contract
        - LabourDay              -float data type, contains the number labourday generated for the contract     
        - StatusID               -float data type, contains the status id of the contract
        - ProjectID              -float data type, contains the project id of the contract
        - ContractTypeID         -float data type, contains the contract type of the contract,
                                                   2 is  Rehabilitatioin and 6 is construction 
        - Project                -string data type, contains the project of the contract i.e. ARAP or NERAP
        - SurfaceOption          -string data type, contains the surfaceoption of the contract intervention 
                                                    such as asphalt, graval, bridge


    """
    df = pd.read_csv('NRAP_dataset.csv', index_col='ContractID')
    return df

def load_shapefile():
    """
    The shapefile provided contains data for 32 provinces in Afghanistan. 
    However, as of March 23, 2023, Afghanistan officially has 34 provinces. Unfortunately, I was unable 
    to find a trusted source for a shapefile that includes the updated information. As a result, 
    this shapefile does not represent the data for the missing provinces of Panjsheer and Daykundi on the map.
    
    Loads the shapefile for admin3_poly_32 and returns it as a GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: The shapefile for admin3_poly_32 as a GeoDataFrame.
    """
    shapefile = gpd.read_file('admin2_poly_32.shp')
    return shapefile


def remove_columns(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove specified columns from a pandas dataframe if they exist.

    Args:
        dataframe (pd.DataFrame): The dataframe to remove columns from.
        columns (list): A list of column names to remove.

    Returns:
        pd.DataFrame: The modified dataframe with the specified columns removed, or the original dataframe if the specified columns do not exist.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> remove_columns(df, ['A'])
           B
        0  3
        1  4
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('The "dataframe" parameter must be a pandas DataFrame')
    if not isinstance(columns, list):
        raise TypeError('The "columns" parameter must be a list of column names')
    columns_to_remove = [col for col in columns if col in dataframe.columns]
    if columns_to_remove:
        dataframe = dataframe.drop(columns_to_remove, axis=1)
    return dataframe


def change_datatype(dataframe: pd.DataFrame, columns: list, datatype: str) -> pd.DataFrame:
    """
    Change the datatype of specified columns in a pandas dataframe if they exist.

    Args:
        dataframe (pd.DataFrame): The dataframe to modify.
        columns (list): A list of column names to modify.
        datatype (str): The datatype to convert the columns to.

    Returns:
        pd.DataFrame: The modified dataframe with the specified columns converted to the specified datatype, 
        or the original dataframe if the specified columns do not exist.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> change_datatype(df, ['A'], 'float')
             A  B
        0  1.0  3
        1  2.0  4
    """
    columns_to_change = [col for col in columns if col in dataframe.columns]
    if columns_to_change:
        dataframe[columns_to_change] = dataframe[columns_to_change].astype(datatype)
    return dataframe


import pandas as pd

def check_for_missing_values(dataframe: pd.DataFrame) -> None:
    """
    Checks a pandas DataFrame for missing values and prints the number of missing values in each column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to check for missing values.

    Raises:
        TypeError: If the input is not a pandas DataFrame.

    Returns:
        None
    """
    # Check that the input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Check for missing values using isnull() or isna()
    null_mask = dataframe.isnull()

    # Count the number of missing values in each column using sum()
    null_counts = null_mask.sum()

    # Print the number of missing values in each column
    if (null_counts > 0).any():
        print("The following columns contain missing values:")
        print(null_counts[null_counts > 0])
    else:
        print("No missing values found in the DataFrame.")
import pandas as pd


def replace_null_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces null values with 0 for specified columns in a pandas DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame to replace null values in.
        
    Returns:
        pandas DataFrame: The modified DataFrame with null values replaced by 0.


    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")


    for col in df.columns:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].fillna(0)
    return df

def show_null_rows(df: pd.DataFrame, col_name: str) -> str:
    """
    Returns the rows of a Pandas DataFrame where a specific column has null values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to check for null values.

    Returns:
        str: A message indicating the number of null rows found in the specified column, or a message
        indicating that no null rows were found.

    Raises:
        ValueError: If the specified column name is not found in the DataFrame.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

    null_mask = df[col_name].isnull()
    null_rows = df[null_mask]

    if null_rows.empty:
        return f"No null values found in column '{col_name}'."

    grouped_df = null_rows.groupby('Status').count()
    num_null_rows = grouped_df.shape[0]

    return f"Found {num_null_rows} null rows in column '{col_name}'."


def fill_nulls(data: pd.DataFrame, null_column: str, fill_column: str) -> pd.DataFrame:
    """
    Fill null values in a column of a pandas DataFrame with values from another column.

    Args:
        data (pd.DataFrame): The DataFrame containing the columns to fill.
        null_column (str): The name of the column containing the null values to fill.
        fill_column (str): The name of the column containing the values to use for filling.

    Returns:
        pd.DataFrame: The modified DataFrame with null values filled in the specified column.

    Raises:
        ValueError: If either the null_column or fill_column is not found in the DataFrame.
    """
    if null_column not in data.columns:
        raise ValueError(f"Column '{null_column}' not found in DataFrame.")

    if fill_column not in data.columns:
        raise ValueError(f"Column '{fill_column}' not found in DataFrame.")

    num_nulls = data[null_column].isnull().sum()

    data[null_column] = data[null_column].fillna(data[fill_column])

    if num_nulls > 0:
        print(f"{num_nulls} null values in '{null_column}' column were filled using values from '{fill_column}' column.")
    else:
        print(f"No null values found in '{null_column}' column.")

    return data

def count_rows(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Takes a Pandas DataFrame and a column name as inputs, and returns the count of rows for each value in that column.

    Args:
        df (pd.DataFrame): The DataFrame to count rows from.
        col_name (str): The name of the column to count rows for.

    Returns:
        pd.Series: A Series containing the count of rows for each unique value in the specified column.

    Raises:
        ValueError: If the specified column name is not found in the DataFrame.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

    counts = df.groupby(col_name).size()

    return counts


def replace_values(df: pd.DataFrame, values_to_replace: list, new_value) -> pd.DataFrame:
    """
    This function takes a Pandas DataFrame, a list of values to replace, and a new value to replace them with.
    It replaces all occurrences of the values in the DataFrame with the new value.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        values_to_replace (list): A list of values to replace.
        new_value: The value to replace the values_to_replace with.

    Returns:
        pd.DataFrame: The modified DataFrame.

    Raises:
        ValueError: If the values_to_replace parameter is not a list.
    """
    if not isinstance(values_to_replace, list):
        raise ValueError("The 'values_to_replace' parameter must be a list.")

    df.replace(to_replace=values_to_replace, value=new_value, inplace=True)

    return df


def outliers_count(df: pd.DataFrame, column: str, threshold: float) -> int:
    """
    Counts the number of outliers in a DataFrame column using Z-score.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to check for outliers.
        threshold (float): The Z-score threshold for identifying outliers.

    Returns:
        int: The number of outliers in the column.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"The '{column}' column does not exist in the DataFrame.")

    z_scores = stats.zscore(df[column])
    positive_outliers = df[z_scores > threshold]
    negative_outliers = df[z_scores < -threshold]
    outliers = pd.concat([positive_outliers, negative_outliers])
    outliers_count = outliers.shape[0]
    
    return outliers_count

def outliers_details(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """
    Finds and returns the details of the outliers in a DataFrame column using Z-score.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name to check for outliers.
        threshold (float): The Z-score threshold for identifying outliers.
        
    Returns:
        pandas.DataFrame: The details of the outliers in the column.
    """
    z_scores = stats.zscore(df[column])
    positive_outliers = df[z_scores > threshold]
    negative_outliers = df[z_scores < -threshold]
    outliers = pd.concat([positive_outliers, negative_outliers])
    return outliers

def plot_outliers(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Creates a box plot of a DataFrame column with outliers highlighted.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        
    Returns:
        None
    """
    # Define box, whisker, cap, flier and median properties with different values
    boxprops = dict(linewidth=2, color='blue')
    whiskerprops = dict(linewidth=2, color='red')
    capprops = dict(linewidth=2, color='green')
    flierprops = dict(marker='o', markerfacecolor='purple', markersize=8, linestyle='none')
    medianprops = dict(linewidth=2, color='orange')

    # Add labels to the x-axis, y-axis and title of the boxplot
    plt.figure(figsize=(10, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Create a boxplot with the previously defined properties
    plt.boxplot(df[column], boxprops=boxprops, whiskerprops=whiskerprops, 
                capprops=capprops, flierprops=flierprops, medianprops=medianprops)

    # Display the boxplot
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt


def plot_outliers_by_category(df: pd.DataFrame, column: str, category_col: str,
                              title: str, xlabel: str, ylabel: str) -> None:
    """
    Creates a box plot of a DataFrame column with outliers highlighted, 
    grouped by a category column.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name to plot.
        category_col (str): The column name to group the plot by.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        
    Returns:
        None
    """
    # Define box, whisker, cap, flier and median properties with different values
    boxprops = dict(linewidth=2, color='blue')
    whiskerprops = dict(linewidth=2, color='red')
    capprops = dict(linewidth=2, color='green')
    flierprops = dict(marker='o', markerfacecolor='purple', markersize=8, linestyle='none')
    medianprops = dict(linewidth=2, color='orange')

    # Create a list of unique categories in the category column
    categories = df[category_col].unique()

    # Create a list to store the data for each category
    data = []
    for category in categories:
        data.append(df[df[category_col] == category][column])

    # Create a figure and set the size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a boxplot with the previously defined properties
    ax.boxplot(data, boxprops=boxprops, whiskerprops=whiskerprops, 
               capprops=capprops, flierprops=flierprops, medianprops=medianprops)

    # Set the tick labels on the x-axis to the categories
    ax.set_xticklabels(categories)

    # Add labels to the x-axis, y-axis and title of the boxplot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Display the boxplot
    plt.show()


def plot_histo(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str) -> None:

    """
    Creates a box plot of a DataFrame column with outliers highlighted.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    # Generate the histogram with 20 bins
    plt.hist(df[column], bins=20, ec='darkblue', color='lightblue')
    # Set the x-axis label
    plt.xlabel(xlabel)
    # Set the y-axis label
    plt.ylabel(ylabel)
    # Set the Title
    plt.title(title)
    # Display the chart
    plt.show()
    
# def get_central_tendency(data):
#     float_cols = data.select_dtypes(include=['float']).columns
#     central_tendency = pd.DataFrame(columns=['Column Name', 'Mean', 'Median', 'Mode', 'Interpretation'])
    
#     for col in float_cols:
#         col_mean = round(data[col].mean(),2)
#         col_median = round(data[col].median(),2)
#         col_mode = stats.mode(data[col]).mode[0]
        
#         interpretation = "Left Skew" if col_mean < col_median else "Right Skew"
        
#         central_tendency = pd.concat([central_tendency,
#                                       pd.DataFrame({'Column Name': col, 'Mean': col_mean, 'Median': col_median,
#                                                     'Mode': col_mode, 'Interpretation': interpretation},
#                                                    index=[0])], ignore_index=True)
    
#     return central_tendency


def get_central_tendency(df):
    """
    Calculates the central tendency of all the float columns in a given pandas DataFrame.
    The function returns a pandas DataFrame with columns for column name, mean, median, mode,
    and interpretation of skewness.
    
    Args:
    data: A pandas DataFrame containing numeric values.
    
    Returns:
    A pandas DataFrame with central tendency measures of all float columns in the input DataFrame.
    """
    
    # Select float columns
    float_cols = df.select_dtypes(include=['float']).columns
    
    # Create a pandas DataFrame to store the results
    central_tendency = pd.DataFrame(columns=['Column Name', 'Mean', 'Median', 'Mode', 'Interpretation'])
    
    # Loop over each float column to calculate central tendency measures
    for col in float_cols:
        col_mean = round(df[col].mean(),2)
        col_median = round(df[col].median(),2)
        col_mode = stats.mode(df[col]).mode[0]
        
        # Determine the interpretation of skewness
        interpretation = "Left Skew" if col_mean < col_median else "Right Skew"
        
        # Append the results to the central_tendency DataFrame
        central_tendency = pd.concat([central_tendency,
                                      pd.DataFrame({'Column Name': col, 'Mean': col_mean, 'Median': col_median,
                                                    'Mode': col_mode, 'Interpretation': interpretation},
                                                   index=[0])], ignore_index=True)
    
    # Return the central tendency DataFrame
    return central_tendency
