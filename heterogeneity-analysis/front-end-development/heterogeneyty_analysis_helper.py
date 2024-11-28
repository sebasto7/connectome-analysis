# -*- coding: utf-8 -*-
"""

Helper file cotaining custom functions
Clean code for publication

@author: Sebastian Molina-Obando
"""

#%% 
#Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, levene,PermutationMethod
from itertools import combinations
import math

#%% 
#Functions
def determine_subgroup(index, dataset_subgroups):
    """
    Determine the subgroup to which a given index belongs.

    Args:
        index (str): The index string to be checked against subgroups.
        dataset_subgroups (list of str): List of possible subgroups.

    Returns:
        str or None: The subgroup found in the index, or None if no match is found.
    """
    for subgroup in dataset_subgroups:
        if subgroup in index:
            return subgroup
    return None


def filter_values(val):
    """
    Format a value to three decimal places if it is less than 0.05; otherwise, return an empty string.

    Args:
        val (float): The value to be formatted.

    Returns:
        str: Formatted string with three decimal places if the value is < 0.05, otherwise an empty string.
    """
    return f"{val:.3f}" if val < 0.05 else ""


def create_column_c(row, A, B):
    """
    Create a new column value based on conditions applied to columns A and B.

    Args:
        row (pd.Series): The row of the DataFrame.
        A (str): The column name A in the DataFrame.
        B (str): The column name B in the DataFrame.

    Returns:
        str: The value for the new column 'C' based on the conditions.
    """
    if row[B] != 0.0:
        return 'None'
    elif row[B] == '':
        return ''
    else:
        return row[A]


def roundup(x):
    """
    Round a number up to the nearest multiple of 10.

    Args:
        x (float): The number to be rounded.

    Returns:
        int: The number rounded up to the nearest multiple of 10.
    """
    return math.ceil(x / 10.0) * 10


def add_n_labels(box, cluster_arrays, df_cluster):
    """
    Add 'N' labels inside each boxplot to indicate the number of data points.

    Args:
        box (matplotlib.axes.Axes): The Axes object where the boxplot is drawn.
        cluster_arrays (dict): Dictionary with cluster names as keys and cluster data as values.
        df_cluster (pd.DataFrame): DataFrame containing the data for each cluster.

    Returns:
        None
    """
    for i, (cluster_name, cluster_values) in enumerate(cluster_arrays.items()):
        num_data_points = len(cluster_values)
        x_pos = i + 1
        y_pos = df_cluster[cluster_name].median()
        box.text(x_pos, y_pos, f'N = {num_data_points}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    box.grid(False)
    box.spines['right'].set_visible(False)
    box.spines['top'].set_visible(False)


def remove_outliers(df, multiplier=1.5):
    """
    Remove outliers from a DataFrame based on the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame from which to remove outliers.
        multiplier (float, optional): The IQR multiplier to determine outlier thresholds. Default is 1.5.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    df_filtered = df[~((df < (q1 - multiplier * iqr)) | (df > (q3 + multiplier * iqr))).any(axis=1)]
    return df_filtered


def replace_outliers_with_nan(df, multiplier=1.5):
    """
    Replace outliers in a DataFrame with NaN based on the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame in which to replace outliers.
        multiplier (float, optional): The IQR multiplier to determine outlier thresholds. Default is 1.5.

    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by NaN.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    df_filtered = df.mask((df < lower_bound) | (df > upper_bound))
    return df_filtered


def perform_levene_test(col1, col2, column_combinations):
    """
    Perform Levene's test for homogeneity of variances between two columns.

    Args:
        col1 (pd.Series): The first column of data.
        col2 (pd.Series): The second column of data.
        column_combinations (list of tuple): List of column pairs to apply Bonferroni correction.

    Returns:
        float: The p-value after applying Bonferroni correction.
    """
    levene_test_results = levene(col1.dropna(), col2.dropna())
    corrected_p_value = levene_test_results.pvalue * len(column_combinations)
    return corrected_p_value


def permutation_test(cluster_df, dataset_df, column1_name, column2_name, num_permutations, seed=None):
    """
    Perform a permutation test to compare the correlation between two columns in different DataFrames.

    Args:
        cluster_df (pd.DataFrame): DataFrame containing the cluster data.
        dataset_df (pd.DataFrame): DataFrame containing the full dataset.
        column1_name (str): The name of the first column to compare.
        column2_name (str): The name of the second column to compare.
        num_permutations (int): The number of permutations to perform.
        seed (int, optional): Seed for random number generator to ensure reproducibility.

    Returns:
        tuple: Observed correlation, p-value from permutation test, and list of shuffled correlations.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset_df_sampled = dataset_df.sample(n=len(cluster_df), replace=False)
    observed_corr = cluster_df[column1_name].corr(cluster_df[column2_name])
    shuffled_corrs = []

    for _ in range(num_permutations):
        shuffled_values = dataset_df_sampled[column2_name].sample(frac=1).values
        shuffled_df = pd.DataFrame({column1_name: cluster_df[column1_name].values,
                                    f"Shuffled_{column2_name}": shuffled_values})
        shuffled_corr = shuffled_df[column1_name].corr(shuffled_df[f"Shuffled_{column2_name}"])
        shuffled_corrs.append(shuffled_corr)

    p_value = (np.sum(np.abs(shuffled_corrs) >= np.abs(observed_corr)) + 1) / (num_permutations + 1)

    return observed_corr, p_value, shuffled_corrs

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations

def calculate_correlation_and_p_values_BH_correction(df):
    """
    Calculate the Pearson correlation coefficients and Benjamini-Hochberg corrected p-values for all pairs of columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for correlation analysis.

    Returns:
        tuple: DataFrames containing the correlation coefficients and the Benjamini-Hochberg corrected p-values.
    """
    correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)
    p_values_correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)
    all_p_values = []
    comparisons = list(combinations(df.columns, 2))

    # Step 1: Calculate correlations and store raw p-values
    for col1, col2 in comparisons:
        x_data, y_data = df[col1], df[col2]
        correlation_coefficient, p_value = pearsonr(x_data, y_data, method=PermutationMethod())
        correlation_df.at[col1, col2] = correlation_coefficient
        correlation_df.at[col2, col1] = correlation_coefficient
        all_p_values.append(p_value)
        p_values_correlation_df.at[col1, col2] = p_value
        p_values_correlation_df.at[col2, col1] = p_value

    np.fill_diagonal(correlation_df.values, 1.0)

    # Step 2: Apply Benjamini-Hochberg correction
    all_p_values = np.array(all_p_values)
    m = len(all_p_values)  # Total number of comparisons
    sorted_indices = np.argsort(all_p_values)
    sorted_p_values = all_p_values[sorted_indices]
    bh_adjusted = np.zeros(m)

    for i, p_val in enumerate(sorted_p_values):
        bh_adjusted[i] = min(p_val * m / (i + 1), 1.0)  # Adjust and cap at 1.0

    # Step 3: Restore the original order of the adjusted p-values
    bh_adjusted_corrected = np.zeros(m)
    bh_adjusted_corrected[sorted_indices] = bh_adjusted

    # Step 4: Fill the p-value matrix with adjusted p-values
    for idx, (col1, col2) in enumerate(comparisons):
        corrected_p_value = round(bh_adjusted_corrected[idx], 4)
        p_values_correlation_df.at[col1, col2] = corrected_p_value
        p_values_correlation_df.at[col2, col1] = corrected_p_value

    return correlation_df, p_values_correlation_df


def calculate_correlation_and_p_values(df):
    """lu
    Calculate the Pearson correlation coefficients and Bonferroni-corrected p-values for all pairs of columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for correlation analysis.

    Returns:
        tuple: DataFrames containing the correlation coefficients and the Bonferroni-corrected p-values.
    """
    correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)
    p_values_correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)
    num_comparisons = len(list(combinations(df.columns, 2)))

    for col1, col2 in combinations(df.columns, 2):
        x_data, y_data = df[col1], df[col2]
        correlation_coefficient, p_value = pearsonr(x_data, y_data, method=PermutationMethod()) 
        correlation_df.at[col1, col2] = correlation_coefficient
        correlation_df.at[col2, col1] = correlation_coefficient
        p_value_corrected = min(p_value * num_comparisons, 1.0)
        p_values_correlation_df.at[col1, col2] = round(p_value_corrected, 4)
        p_values_correlation_df.at[col2, col1] = round(p_value_corrected, 4)

    np.fill_diagonal(correlation_df.values, 1.0)

    return correlation_df, p_values_correlation_df

def cosine_similarity_and_clustering(_data, cosine_subgroups):
    """
    Perform cosine similarity analysis and hierarchical clustering on the given data.

    Args:
        _data (pd.DataFrame): The DataFrame containing the data for analysis.
        cosine_subgroups (list of str): List of subgroups to calculate cosine similarity within and between.

    Returns:
        tuple: DataFrames for cosine similarity, summary, reordered data, and other clustering results.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.cluster import hierarchy

    # Drop rows with all NaN values
    dropped_indexes = []
    kept_indexes = []
    dropped_data = _data.dropna(how='all', inplace=False)
    dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
    kept_indexes.extend(dropped_data.index)
    print(f"Dropping {len(dropped_indexes)} rows with no data during cosine_sim analysis")

    _data.dropna(how='all', inplace=True)

    if _data.empty:
        raise ValueError("The input DataFrame is empty after removing rows with all NaN values.")

    # Create subgroup data
    subgroup_data = {}
    for subgroup in cosine_subgroups:
        subgroup_df = _data[_data.index.str.contains(subgroup)]
        if subgroup_df.empty:
            raise ValueError(f"No data found for subgroup '{subgroup}'. Check the input DataFrame or filtering condition.")
        subgroup_data[subgroup] = subgroup_df

    # Calculate cosine similarity within subgroups
    cos_sim_within = {}
    cos_sim_within_medians = {}
    for subgroup, subgroup_df in subgroup_data.items():
        if not subgroup_df.empty:
            cos_sim_within[subgroup] = cosine_similarity(subgroup_df.fillna(0))
            cos_sim_within_medians[subgroup] = list(np.round(np.nanmedian(cos_sim_within[subgroup], axis=1), 2))
        else:
            cos_sim_within[subgroup] = np.array([])
            cos_sim_within_medians[subgroup] = []

    # Calculate cosine similarity between subgroups
    cos_sim_between = cosine_similarity(
        subgroup_data[cosine_subgroups[0]].fillna(0), 
        subgroup_data[cosine_subgroups[1]].fillna(0)
    )
    cos_sim_between_medians = list(np.round(np.nanmedian(cos_sim_between, axis=1), 2))

    # Combine cosine similarity medians
    cos_sim_medians = cos_sim_within_medians
    cos_sim_medians[''.join(cosine_subgroups)] = cos_sim_between_medians

    # Fill remaining NaN values in the entire dataset
    _data.fillna(0, inplace=True)

    # Calculate full cosine similarity
    cosine_sim = cosine_similarity(_data.values)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=_data.index, columns=_data.index)

    # Extract features for summary
    hemisphere_list = [index_name.split(':')[2][0] for index_name in _data.index]
    d_v_list = [index_name.split(':')[3] for index_name in _data.index]
    cell_type_list = [index_name.split(':')[0] for index_name in _data.index]

    # Create cosine similarity summary DataFrame
    cosine_sim_summary_df = pd.DataFrame({
        'cosine_sim': np.round(np.nanmedian(np.where(cosine_sim == 1., np.nan, cosine_sim), axis=1), 2),
        'dorso-ventral': d_v_list,
        'hemisphere': hemisphere_list,
        'neuron': cell_type_list
    }, index=_data.index)

    # Perform hierarchical clustering
    dendrogram_cosine = hierarchy.linkage(cosine_sim, method='ward')
    cosine_row_order = hierarchy.leaves_list(dendrogram_cosine)

    # Reorder the data based on clustering
    _data_reordered_cosine_sim = _data.iloc[cosine_row_order].copy()
    cosine_sim_reordered = cosine_similarity(_data_reordered_cosine_sim.values)
    cosine_sim_reordered_df = pd.DataFrame(
        cosine_sim_reordered,
        index=_data_reordered_cosine_sim.index,
        columns=_data_reordered_cosine_sim.index
    )

    return (cosine_sim_df, cosine_sim_summary_df, cosine_row_order, dendrogram_cosine, 
            cosine_sim_reordered_df, _data_reordered_cosine_sim, cosine_sim, cosine_sim_reordered, cos_sim_medians)



# def cosine_similarity_and_clustering(_data, cosine_subgroups):
#     """
#     Perform cosine similarity analysis and hierarchical clustering on the given data.

#     Args:
#         _data (pd.DataFrame): The DataFrame containing the data for analysis.
#         cosine_subgroups (list of str): List of subgroups to calculate cosine similarity within and between.

#     Returns:
#         tuple: DataFrames for cosine similarity, summary, reordered data, and other clustering results.
#     """
#     import numpy as np
#     import pandas as pd
#     from sklearn.metrics.pairwise import cosine_similarity
#     from scipy.cluster import hierarchy

#     dropped_indexes = []
#     kept_indexes = []
#     dropped_data = _data.dropna(how='all', inplace=False)
#     dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
#     kept_indexes.extend(dropped_data.index)
#     print(f'Dropping {len(dropped_indexes)} Tm9 columns with no data during cosine_sim analysis')
#     _data.dropna(how='all', inplace=True)

#     subgroup_data = {}
#     for subgroup in cosine_subgroups:
#         subgroup_data[subgroup] = _data[_data.index.str.contains(subgroup)]

#     cos_sim_within = {}
#     cos_sim_within_medians = {}
#     for subgroup, subgroup_df in subgroup_data.items():
#         cos_sim_within[subgroup] = cosine_similarity(subgroup_df.fillna(0))
#         cos_sim_within_medians[subgroup] = list(np.round(np.nanmedian(cos_sim_within[subgroup], 1), 2))

#     cos_sim_between = cosine_similarity(subgroup_data[cosine_subgroups[0]].fillna(0), subgroup_data[cosine_subgroups[1]].fillna(0))
#     cos_sim_between_medians = list(np.round(np.nanmedian(cos_sim_between, 1), 2))

#     cos_sim_medians = cos_sim_within_medians
#     cos_sim_medians[''.join(cosine_subgroups)] = cos_sim_between_medians

#     _data.fillna(0, inplace=True)

#     cosine_sim = cosine_similarity(_data.values)
#     cosine_sim_df = pd.DataFrame(cosine_sim, index=_data.index, columns=_data.index)

#     hemisphere_list = [index_name.split(':')[2][0] for index_name in _data.index]
#     d_v_list = [index_name.split(':')[3] for index_name in _data.index]
#     cell_type_list = [index_name.split(':')[0] for index_name in _data.index]

#     cosine_sim_summary_df = pd.DataFrame(columns=['cosine_sim', 'dorso-ventral', 'hemisphere','neuron'],
#                                          index=_data.index.tolist())
#     cosine_sim_nan = np.where(cosine_sim == 1., np.nan, cosine_sim)
#     cosine_sim_list = np.round(np.nanmedian(cosine_sim_nan, 1), 2)
#     cosine_sim_summary_df['cosine_sim'] = cosine_sim_list
#     cosine_sim_summary_df['hemisphere'] = hemisphere_list
#     cosine_sim_summary_df['dorso-ventral'] = d_v_list
#     cosine_sim_summary_df['neuron'] = cell_type_list

#     dendrogram_cosine = hierarchy.linkage(cosine_sim, method='ward')
#     cosine_row_order = hierarchy.leaves_list(dendrogram_cosine)

#     _data_reordered_cosine_sim = _data.iloc[cosine_row_order].copy()

#     cosine_sim_reordered = cosine_similarity(_data_reordered_cosine_sim.values)
#     cosine_sim_reordered_df = pd.DataFrame(cosine_sim_reordered,
#                                            index=_data_reordered_cosine_sim.index,
#                                            columns=_data_reordered_cosine_sim.index)

#     return (cosine_sim_df, cosine_sim_summary_df, cosine_row_order, dendrogram_cosine, 
#             cosine_sim_reordered_df, _data_reordered_cosine_sim, cosine_sim, cosine_sim_reordered, cos_sim_medians)
