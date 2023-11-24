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
from scipy.stats import pearsonr, levene
from itertools import combinations
import math

#%% 
#Functions
def determine_subgroup(index, dataset_subgroups):
    for subgroup in dataset_subgroups:
        if subgroup in index:
            return subgroup
    return None

def filter_values(val):
        return f"{val:.3f}" if val < 0.05 else ""

def create_column_c(row, A, B):
    if row[B] != 0.0:
        return 'None'
    elif row[B] == '':
        return ''
    else:
        return row[A]

def roundup(x):
    return math.ceil(x / 10.0) * 10
    
# Function to add N labels inside each boxplot
def add_n_labels(box, cluster_arrays, df_cluster):
    for i, (cluster_name, cluster_values) in enumerate(cluster_arrays.items()):
        # Get the number of data points (N) for each boxplot
        num_data_points = len(cluster_values)

        # Calculate the position to place the text inside the boxplot
        x_pos = i + 1
        y_pos = df_cluster[cluster_name].median()  # Y position inside the box is set to the median of the data

        # Add the N label inside the boxplot
        box.text(x_pos, y_pos, f'N = {num_data_points}', ha='center', va='center', fontsize=10, fontweight='bold')
        
    box.grid(False)
    # Remove the left and upper border lines
    box.spines['right'].set_visible(False)
    box.spines['top'].set_visible(False)
    
def remove_outliers(df, multiplier=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    
    # Calculate the IQR for each column
    iqr = q3 - q1
    
    # Filter out rows where any column has a value outside the range [Q1 - multiplier * IQR, Q3 + multiplier * IQR]
    df_filtered = df[~((df < (q1 - multiplier * iqr)) | (df > (q3 + multiplier * iqr))).any(axis=1)]
    
    return df_filtered

def replace_outliers_with_nan(df, multiplier=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    # Calculate the IQR for each column
    iqr = q3 - q1

    # Determine the lower and upper bounds for outlier detection
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Replace outlier values with NaN
    df_filtered = df.mask((df < lower_bound) | (df > upper_bound))

    return df_filtered

def perform_levene_test(col1, col2, column_combinations):
    levene_test_results = levene(col1.dropna(), col2.dropna())
    # Bonferroni correction
    corrected_p_value = levene_test_results.pvalue * len(column_combinations)
    return corrected_p_value

def permutation_test(cluster_df, dataset_df, column1_name, column2_name, num_permutations, seed= None):
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    #print(f'Using seed: {seed} for random selection of optic lobe columns from the full data set') 
    # dataset_df = dataset_df.fillna(0).copy()  
    # cluster_df = cluster_df.fillna(0).copy()   

    # Randomly select the same number of rows from dataset_df as in cluster_df
    dataset_df_sampled = dataset_df.sample(n=len(cluster_df), replace=False)

    observed_corr = cluster_df[column1_name].corr(cluster_df[column2_name])  # Compute the observed correlation
    shuffled_corrs = []

    for _ in range(num_permutations):
        shuffled_values = dataset_df_sampled[column2_name].sample(frac=1).values  # Shuffle the values of the second column
        shuffled_df = pd.DataFrame({column1_name: cluster_df[column1_name].values,
                                    f"Shuffled_{column2_name}": shuffled_values})
        shuffled_corr = shuffled_df[column1_name].corr(shuffled_df[f"Shuffled_{column2_name}"])
        shuffled_corrs.append(shuffled_corr)

    # Calculate the p-value based on the number of shuffled correlations larger or equal to the observed correlation
    p_value = (np.sum(np.abs(shuffled_corrs) >= np.abs(observed_corr)) + 1) / (num_permutations + 1)

    return observed_corr, p_value, shuffled_corrs

def calculate_correlation_and_p_values(df):
    # Initialize empty DataFrames for correlation and p-values
    correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)
    p_values_correlation_df = pd.DataFrame(columns=df.columns, index=df.columns)

    num_comparisons = len(list(combinations(df.columns, 2)))

    # Calculate the correlation matrix using Pearson correlation
    for col1, col2 in combinations(df.columns, 2):
        # Get the data for the current pair of columns
        x_data, y_data = df[col1], df[col2]

        # Compute the Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(x_data, y_data)

        # Store the absolute value of the correlation coefficient in the DataFrame
        correlation_df.at[col1, col2] = correlation_coefficient
        correlation_df.at[col2, col1] = correlation_coefficient

        # Store the p-value adjusted with Bonferroni correction in the DataFrame
        p_value_corrected = min(p_value * num_comparisons, 1.0)
        p_values_correlation_df.at[col1, col2] = round(p_value_corrected, 4)
        p_values_correlation_df.at[col2, col1] = round(p_value_corrected, 4)

    # Fill the diagonal with 1.0 since the correlation of a variable with itself is always 1
    np.fill_diagonal(correlation_df.values, 1.0)

    return correlation_df, p_values_correlation_df


def cosine_similarity_and_clustering(_data,cosine_subgroups):
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.cluster import hierarchy
    # Filtering out columns with no data
    dropped_indexes = []
    kept_indexes = []
    dropped_data = _data.dropna(how='all', inplace=False)
    dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
    kept_indexes.extend(dropped_data.index)
    print(f'Dropping {len(dropped_indexes)} Tm9 columns with no data during cosine_sim analysis')
    _data.dropna(how='all', inplace=True)  # now dropping if all values in the row are nan

    #Doing cosine similarities in subgroups in the data set
    # Separate data into subgroups based on subgroup letters in the index
    subgroup_data = {}
    for subgroup in cosine_subgroups:
        subgroup_data[subgroup] = _data[_data.index.str.contains(subgroup)]

    # Calculate cosine similarity within each subgroup
    cos_sim_within = {}
    cos_sim_within_medians = {}
    for subgroup, subgroup_df in subgroup_data.items():
        cos_sim_within[subgroup] = cosine_similarity(subgroup_df.fillna(0))
        cos_sim_within_medians[subgroup] = list(np.round(np.nanmedian(cos_sim_within[subgroup], 1), 2)) # pulling values together for each postsynaptic neuron

    # Calculate cosine similarity between subgroups if needed
    cos_sim_between = cosine_similarity(subgroup_data[cosine_subgroups[0]].fillna(0), subgroup_data[cosine_subgroups[1]].fillna(0))
    cos_sim_between_medians = list(np.round(np.nanmedian(cos_sim_between, 1), 2)) # pulling values together for each postsynaptic neuron

    # Within and between together in a dictionary
    cos_sim_medians = cos_sim_within_medians
    cos_sim_medians[''.join(cosine_subgroups)] = cos_sim_between_medians

    _data.fillna(0, inplace=True)  # Filling the remaining absent connectivity with a meaningful zero

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(_data.values)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=_data.index, columns=_data.index)


    hemisphere_list = [index_name.split(':')[2][0] for index_name in _data.index]
    d_v_list = [index_name.split(':')[3] for index_name in _data.index]
    cell_type_list = [index_name.split(':')[0] for index_name in _data.index]

    cosine_sim_summary_df = pd.DataFrame(columns=['cosine_sim', 'dorso-ventral', 'hemisphere','neuron'],
                                         index=_data.index.tolist())
    cosine_sim_nan = np.where(cosine_sim == 1., np.nan, cosine_sim)
    cosine_sim_list = np.round(np.nanmedian(cosine_sim_nan, 1), 2) # pulling values together for each postsynaptic neuron
    cosine_sim_summary_df['cosine_sim'] = cosine_sim_list
    cosine_sim_summary_df['hemisphere'] = hemisphere_list
    cosine_sim_summary_df['dorso-ventral'] = d_v_list
    cosine_sim_summary_df['neuron'] = cell_type_list


    dendrogram_cosine = hierarchy.linkage(cosine_sim, method='ward')
    cosine_row_order = hierarchy.leaves_list(dendrogram_cosine)

    _data_reordered_cosine_sim = _data.iloc[cosine_row_order].copy()

    cosine_sim_reordered = cosine_similarity(_data_reordered_cosine_sim.values)
    cosine_sim_reordered_df = pd.DataFrame(cosine_sim_reordered,
                                          index=_data_reordered_cosine_sim.index,
                                          columns=_data_reordered_cosine_sim.index)

    return cosine_sim_df, cosine_sim_summary_df, cosine_row_order, dendrogram_cosine, cosine_sim_reordered_df, _data_reordered_cosine_sim, cosine_sim, cosine_sim_reordered, cos_sim_medians