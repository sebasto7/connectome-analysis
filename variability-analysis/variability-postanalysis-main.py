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
from scipy.stats import kruskal, bartlett
from scikit_posthocs import posthoc_dunn
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


#Importing custom functions from helper file
from helper import cosine_similarity_and_clustering, remove_outliers, perform_levene_test, determine_subgroup

#%% Custom functions 
#TODO load this functions from a helper file at some point




#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################

# Specify the folder containing files (processed-data)
PC_disc = 'D'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data' # Path to the PROCESSED_DATA folder
fig_save_path = os.path.join(dataPath,"Figures")
save_figures = True
exclude_outliers = False


# Comparisons (between processed-data)
single_data_set = False # True, False
data_frames_to_compare_ls = ['Tm9_FAFB_R_', 'Tm1_FAFB_R_','Tm2_FAFB_R_'] # ['Tm9_300_healthy_L3_L_R_20230823'], ['Tm9_FAFB_R_'], ['Tm9_FAFB_L_R_'], ['Tm9_FAFB_R_', 'Tm1_FAFB_R_','Tm2_FAFB_R_']
user_defined_categoriers = ['Tm9_R', 'Tm1_R', 'Tm2_R'] # ['Tm9_R'] , ['Tm9_R', 'Tm1_R', 'Tm2_R']
dataset_subgroups = ['R', 'L'] # ['D', 'V'], ['R', 'L']
dataset = ['FAFB_R_L'] # ['FAFB_R_L']
subgroups_name = 'dorso-ventral' # 'dorso-ventral', hemisphere


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


############################################# Synapse count variation #########################################

## Synapse count distributions for ABSOLUTE COUNTS

# Initialize an empty DataFrame
syn_count_df = pd.DataFrame()

# Find the maximum length among all lists
max_length = max(len(_data) for _data in data_frames.values())

# Iterate over each DataFrame
for i, df_name in enumerate(data_frames_to_compare_ls):
    df_name = df_name + '_Absolut_counts'
    _data = data_frames[df_name]
    
    # Sum all columns along the rows to get 'total_count'
    _data['total_count'] = _data.sum(axis=1)
    
    # Add a new column with NaN values if the length is less than the maximum length
    syn_count_df[user_defined_categoriers[i]] = _data['total_count'].tolist() + [np.nan] * (max_length - len(_data['total_count']))


# ##################################    Cosine similarity in absolute counts    #################################

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


## For single data set
if single_data_set:
    rel_df = data_frames[data_frames_to_compare_ls[0]+'_Relative_counts']
    rel_df.fillna(0, inplace=True)
    
    # Apply the function to create the "dorso-ventral" column
    rel_df[subgroups_name] = rel_df.index.map(lambda x: determine_subgroup(x, dataset_subgroups))



#%% 
############################################ PLOTS and STATISTICS #############################################
###############################################################################################################


########################################### Synapse count variability ########################################
##############################    Leven test for equality of variances    ####################################



# Plotting
# Plot box plots and histograms in two subplots
_binwidth = 6
# Removing outliers
if exclude_outliers:
    syn_count_df = remove_outliers(syn_count_df, multiplier=1.5)

# Calculate the coefficient of variation (CV) for each column
cv_values = syn_count_df.std() / syn_count_df.mean()

# Perform F-test for equality of variances
f_test_results = f_oneway(*[syn_count_df[col].dropna() for col in syn_count_df.columns])

# Perform Bartlett's test for equality of variances
bartlett_test_results = bartlett(*[syn_count_df[col].dropna() for col in syn_count_df.columns])

# Perform Levene's test for equality of variances pairwise with Bonferroni correction
column_combinations = list(combinations(syn_count_df.columns, 2))
alpha = 0.05  # Set your desired significance level

# Create subplots for box plots and histograms
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plots with the same colors used in histograms
sns.boxplot(data=syn_count_df, ax=axes[0], palette=sns.color_palette('husl', n_colors=len(syn_count_df.columns)))
axes[0].set_title("Synapse count variability (Levene's Test)")
axes[0].set_ylabel('Synapse counts')

# Add CV values to the box plots
for i, col in enumerate(syn_count_df.columns):
    axes[0].text(i, syn_count_df[col].max() + 10, f'CV={cv_values[col]:.2f}', ha='center', va='bottom', color='blue')

# Plot horizontal lines with p-values
for i, (col1, col2) in enumerate(column_combinations):
    p_value = perform_levene_test(syn_count_df[col1], syn_count_df[col2],column_combinations)

    print(f"Levene's Test for {col1} and {col2} p-value (Bonferroni corrected): {p_value:.4f}")
    print("Significant" if p_value < alpha else "Not significant")

    # Extract x-axis tick locations for each column
    ticks = axes[0].get_xticks()
    
    # Find the index of the current columns in the list of ticks
    index_col1 = syn_count_df.columns.get_loc(col1)
    index_col2 = syn_count_df.columns.get_loc(col2)
    
    # Calculate the center positions based on the tick locations
    center1 = ticks[index_col1]
    center2 = ticks[index_col2] 
    
    y_position = max(syn_count_df[col1].max(), syn_count_df[col2].max()) + 20
    
    # Plot horizontal lines from one boxplot center to the other
    axes[0].hlines(y=y_position, xmin=center1, xmax=center2, color='red', linewidth=2)
    axes[0].text((center1 + center2) / 2, y_position + 2, f'p={p_value:.4f}', ha='center', va='bottom', color='red')

# Histograms for each column without outliers using Seaborn with the same colors
for col_idx, (col, color) in enumerate(zip(syn_count_df.columns, sns.color_palette('husl', n_colors=len(syn_count_df.columns)))):
    sns.histplot(data=syn_count_df[col], binwidth=_binwidth, alpha=0.5, ax=axes[1], kde=True, label=col, color=color)

axes[1].set_title('Synapse count variability')
axes[1].set_xlabel('Synapse counts')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Save the figure if required
if save_figures:
    figure_title = '\Synaptic_count_variability_no_ouliers.pdf'
    plt.savefig(fig_save_path + figure_title)




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

if save_figures:
    plt.savefig(f'{fig_save_path}\Cosine_similarity_{dataset}_abs_count.pdf')

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

if save_figures:
    figure_title = f'\Testing_violin_plots.pdf'
    fig.savefig(fig_save_path+figure_title)

print('Coding here')