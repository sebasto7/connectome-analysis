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
from scipy import stats
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


#Importing custom functions from helper file
from helper import cosine_similarity_and_clustering

#%% Custom functions 
#TODO load this functions from a helper file at some point




#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################

# Specify the folder containing files (processed-data)
PC_disc = 'D'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data'
save_figures = True


# Comparisons (between processed-data)
single_data_set = True
data_frames_to_compare_ls = ['Tm9_FAFB_L_R_'] # ['Tm9_FAFB_R_'], ['Tm9_FAFB_L_R_'], ['Tm9_FAFB_R_', 'Tm1_FAFB_R_','Tm2_FAFB_R_']
user_defined_categoriers = ['Tm9_R'] # ['Tm9_R', 'Tm1_R', 'Tm2_R']
dataset_subgroups = ['D', 'V'] # ['D', 'V'], ['R', 'L']
subgroups_name = 'dorso-ventral' # 'dorso-ventral'


excel_file_to_load = []
for file_name in data_frames_to_compare_ls:
    file_name = file_name + '.xlsx'
    excel_file_to_load.append(os.path.join(dataPath,file_name))


#%% 
######################################## LOADING PREPROCESSED DATA ############################################
###############################################################################################################

# Get a list of all Excel files in the folder
excel_files = glob.glob(os.path.join(dataPath , '*.xlsx'))

# Initialize an empty dictionary to store DataFrames
data_frames = {}

# Iterate through each Excel file
for excel_file in excel_file_to_load:
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

##################################    Cosine similarity in absolute counts    #################################


## For multiple data sets
# Computing cosine similarity for absolute counts
combined_cosine_sim_summary_df = pd.DataFrame()
cos_sim_medians_dict = {}
for i,df_name in enumerate(data_frames_to_compare_ls):
    df_name = df_name + '_Absolut_counts'
    _data = data_frames[df_name]
    # Call the function and only define cosine_sim_summary_df
    cosine_sim_summary_df = cosine_similarity_and_clustering(_data,dataset_subgroups)[1]
    cosine_sim_summary_df['neuron'] = user_defined_categoriers[i]
    #Call the function and only define cos_sim_medians 
    cos_sim_medians_dict[df_name] = cosine_similarity_and_clustering(_data,dataset_subgroups)[8]

    # Concatenate the current dataframe to the combined dataframe
    combined_cosine_sim_summary_df = pd.concat([combined_cosine_sim_summary_df, cosine_sim_summary_df])

# Reset index of the combined dataframe
combined_cosine_sim_summary_df = combined_cosine_sim_summary_df.reset_index(drop=True)

## For single data set
if single_data_set:
    _dict = cos_sim_medians_dict[data_frames_to_compare_ls[0]+'_Absolut_counts']

    combined_cosine_sim_list = []
    combined_neuron_list = []
    for key,value in _dict.items():
        combined_cosine_sim_list = combined_cosine_sim_list + value
        combined_neuron_list = combined_neuron_list + [key]*len(value)

    # Create the dataframe with all subgropus
    combined_cosine_sim_summary_df = pd.DataFrame()
    combined_cosine_sim_summary_df['cosine_sim'] = combined_cosine_sim_list
    combined_cosine_sim_summary_df['neuron'] = combined_neuron_list


############################################    Relative counts    #########################################


def determine_subgroup(index, dataset_subgroups):
    for subgroup in dataset_subgroups:
        if subgroup in index:
            return subgroup
    return None


## For single data set
if single_data_set:
    rel_df = data_frames[data_frames_to_compare_ls[0]+'_Relative_counts']
    rel_df.fillna(0, inplace=True)
    
    # Apply the function to create the "dorso-ventral" column
    rel_df[subgroups_name] = rel_df.index.map(lambda x: determine_subgroup(x, dataset_subgroups))



#%% 
############################################ PLOTS and STATISTICS #############################################
###############################################################################################################


############################################    Cosine similarity     #########################################
###########################################    Multiple Comparisons    ########################################

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



############################################    Relative counts    #########################################


##Plotting:

# Filter out the "dorso-ventral" column
data_cols = rel_df.columns[:-1]

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize a list to store violin plot data
violin_plot_data = []

# Create an offset for the dodge effect
offset = 0

# Iterate through each data column
for i, col in enumerate(data_cols):
    # Create a DataFrame for the current data column and "dorso-ventral" column
    data = rel_df[[col, subgroups_name]]
    
    # Create a violin plot for the current data column
    sns.violinplot(x=subgroups_name, y=col, data=data, ax=ax, position=i+offset)
    
    # Update the offset for the next plot
    offset += 0.2  # Adjust the value as needed
    
    # Append the data to the list
    violin_plot_data.append(data)

# Set the x-axis label
ax.set_xlabel(subgroups_name)
ax.set_ylabel('Synaptic count (%)')

# Set the title
ax.set_title("Violin Plots for Each Column")


##Statistics. Pair-wise comparison between subgroups in each cell type
# !!! So far meant for just 2 subgroups only

data_cols = rel_df.columns[:-1]

# Create an empty list to store p-values
p_values_list = []

# Iterate through each data column
for col in data_cols:
    # Get unique dorso-ventral categories
    categories = rel_df[subgroups_name].unique()
    
    # Initialize a list to store p-values for the current data column
    p_values_col = []
    
    # If only one category exists, append None to p_values_col and continue
    if len(categories) == 1:
        p_values_col.append(None)
        p_values_list.append(p_values_col)
        continue
    
    # Generate combinations of categories for pairwise comparison
    category_combinations = combinations(categories, 2)
    
    # Iterate through category combinations
    for cat1, cat2 in category_combinations:
        group1 = rel_df[rel_df[subgroups_name] == cat1][col]
        group2 = rel_df[rel_df[subgroups_name] == cat2][col]
        
        # Perform the Shapiro-Wilk test for normality
        _, p_value1 = stats.shapiro(group1)
        _, p_value2 = stats.shapiro(group2)
        
        # Decide whether to use parametric or non-parametric test based on normality
        if p_value1 > 0.05 and p_value2 > 0.05:
            t_statistic, p_value = stats.ttest_ind(group1, group2)
            print(f'{col} is normally distributed')
        else:
            _, p_value = stats.mannwhitneyu(group1, group2)
        
        p_values_col.append(p_value)
    
    p_values_list = p_values_list + p_values_col

# Convert the p-values list to a DataFrame
p_values_df = pd.DataFrame()
p_values_df['Neuron'] = data_cols.tolist()
p_values_df['p_value'] = p_values_list

# Display the p-values DataFrame
print(f'Significant difference between {subgroups_name}')
print(p_values_df[p_values_df["p_value"]<0.05])


print('Coding here')