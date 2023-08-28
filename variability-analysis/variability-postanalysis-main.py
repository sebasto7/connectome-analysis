# -*- coding: utf-8 -*-
"""
Created on Friday 25 16:40:16 2023

@author: smolina

variability postanalysis of presynaptic inputs
"""
#%% Importing packages
import os
import pandas as pd
import glob


import pandas as pd
import numpy as np
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import matplotlib.pyplot as plt
import seaborn as sns

#%% Custom functions 
#TODO load this functions from a helper file at some point

def cosine_similarity_and_clustering(_data):
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
    cosine_sim_list = np.round(np.nanmedian(cosine_sim_nan, 1), 2)
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

    return cosine_sim_df, cosine_sim_summary_df, cosine_row_order, dendrogram_cosine, cosine_sim_reordered_df, _data_reordered_cosine_sim, cosine_sim,cosine_sim_reordered


#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################

# Specify the folder containing files (processed-data)
PC_disc = 'C'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data'


# Comparisons (between processed-data)
data_frames_to_compare_ls = ['Tm1_FAFB_R__Absolut_counts', 'Tm2_FAFB_R__Absolut_counts', 'Tm9_FAFB_R__Absolut_counts']

#%% 
######################################## LOADING PREPROCESSED DATA ############################################
###############################################################################################################

# Get a list of all Excel files in the folder
excel_files = glob.glob(os.path.join(dataPath , '*.xlsx'))

# Initialize an empty dictionary to store DataFrames
data_frames = {}

# Iterate through each Excel file
for excel_file in excel_files:
    # Get the distinct part of the filename (excluding extension)
    file_name = os.path.splitext(os.path.basename(excel_file))[0]
    
    # Load all sheets from the Excel file into a dictionary of DataFrames
    sheet_dataframes = pd.read_excel(excel_file, sheet_name=None,index_col = 0)
    
    # Iterate through each sheet DataFrame
    for sheet_name, sheet_df in sheet_dataframes.items():
        # Create a key for the combined name of DataFrame
        df_name = f"{file_name}_{sheet_name}"
        
        # Store the DataFrame in the dictionary
        data_frames[df_name] = sheet_df

# # Now you can access the DataFrames by their combined names
# for df_name, df in data_frames.items():
#     print(f"DataFrame Name: {df_name}")
#     print(df)  # Print the DataFrame
#     print("\n")


#%% 
############################################### DATA ANALYSIS #################################################
###############################################################################################################

############################################    Cosine similarity     #########################################

# Computing cosine similarity
data_frames_to_compare_ls = ['Tm1_FAFB_R__Absolut_counts', 'Tm2_FAFB_R__Absolut_counts', 'Tm9_FAFB_R__Absolut_counts']
combined_cosine_sim_summary_df = pd.DataFrame()
for df_name in data_frames_to_compare_ls:
    _data = data_frames[df_name]
    # Call the function and only define cosine_sim_summary_df
    cosine_sim_summary_df = cosine_similarity_and_clustering(_data)[1]

    # Concatenate the current dataframe to the combined dataframe
    combined_cosine_sim_summary_df = pd.concat([combined_cosine_sim_summary_df, cosine_sim_summary_df])

# Reset index of the combined dataframe
combined_cosine_sim_summary_df = combined_cosine_sim_summary_df.reset_index(drop=True)




#%% 
############################################ PLOTS and STATISTICS #############################################
###############################################################################################################


############################################    Cosine similarity     #########################################
data = combined_cosine_sim_summary_df.copy()

### Chekcing data distribution
# Check if the data in each category is normally distributed using the Shapiro-Wilk test:
categories = data["neuron"].unique()
normality_results = {}

for category in categories:
    category_data = data[data["neuron"] == category]["cosine_sim"]
    _, p_value = shapiro(category_data)
    normality_results[category] = p_value

print("Shapiro-Wilk p-values for normality:")
print(normality_results)

### Perform One-Way ANOVA and Multiple Comparisons:
# Perform one-way ANOVA and then use the Tukey HSD test for multiple comparisons if the data is normally distributed:
anova_results = f_oneway(*[data[data["neuron"] == category]["cosine_sim"] for category in categories])

if all(p > 0.05 for p in normality_results.values()):
    print("One-Way ANOVA p-value:", anova_results.pvalue)
    tukey_results = pairwise_tukeyhsd(data["cosine_sim"], data["neuron"])
    print(tukey_results)
else:
    print("Data is not normally distributed. Performing Kruskal-Wallis test.")
    kruskal_results = kruskal(*[data[data["neuron"] == category]["cosine_sim"] for category in categories])
    print("Kruskal-Wallis p-value:", kruskal_results.pvalue)

    # Applying Dunn-Bonferroni correction for multiple comparisons
    dunn_results = posthoc_dunn(data, val_col="cosine_sim", group_col="neuron", p_adjust="bonferroni")
    print(dunn_results)

# Plot box plots with p-values
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="neuron", y="cosine_sim")
plt.title("Cosine Similarity by Neuron Category")
plt.ylabel("Cosine Similarity")
plt.xlabel("Neuron Category")

# Adding p-values to the plot
comparison_results = tukey_results if "tukey_results" in locals() else dunn_results

# Adding lines and p-values to the plot
line_distance = 0.05  # Adjust this value to increase the distance between lines

line_positions = {}  # Store line positions for each comparison

for i, category1 in enumerate(categories):
    for j, category2 in enumerate(categories):
        if j > i:  # Avoid redundant comparisons
            y_pos1 = max(data[data["neuron"] == category1]["cosine_sim"]) + 0.02
            y_pos2 = max(data[data["neuron"] == category2]["cosine_sim"]) + 0.02
            y_line = max(y_pos1, y_pos2) + (line_distance * len(line_positions))
            line_positions[(i, j)] = y_line
            
            # Calculate x position for the line
            x_pos = (i + j) / 2
            
            # Access p-values based on the analysis performed
            if "tukey_results" in locals():
                p_value = tukey_results.pvalues[i, j]
            else:
                p_value = comparison_results.loc[category1, category2]
            
            # Draw line and add p-value text
            plt.plot([i, j], [y_line, y_line], linewidth=1, color='black')
            plt.text(x_pos, y_line, f"p = {p_value:.4f}", ha='center')

# Adjust ylim to fit the lines and p-values
plt.ylim(plt.ylim()[0], max(line_positions.values()) + line_distance)

plt.show()













print('Coding here')