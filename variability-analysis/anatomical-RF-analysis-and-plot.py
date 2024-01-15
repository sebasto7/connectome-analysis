# -*- coding: utf-8 -*-
"""
Anatomical receptive field (RF) analysis and plot, including area and column span.
Clean code for publication

@author: Sebastian Molina-Obando
"""

#%% Importing packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance
from fafbseg import flywire
import navis
from helper_anatomical_RF_analysis import *


#%% General plotting settings
PC_disc = 'D'
dataPath = f'{PC_disc}:\FlyWire-Data\Processed-data'
fig_save_path = os.path.join(dataPath,"Figures")
save_figures = True

#General style
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 8}
axes = {'labelsize': 12, 'titlesize': 12}
ticks = {'labelsize': 10}
legend = {'fontsize': 8}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', **ticks)
plt.rc('ytick', **ticks)

#Saving text for pdfs in a proper way
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# For plotting purposes

hemisphere = 'R' 
neuropile_mesh = 'ME_L' # for fafbseq before version 2.0.0 (if later used, this should read: 'ME_R')
mesh_azim = 16
mesh_elev = -50 
cm = 1/2.54  # centimeters in inches



#%% Data analysis 
##################################################### Presynaptic partner analysis ########################################################
# Loading data
PC_disc = 'D'
dataPath = f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data'
fig_save_path = os.path.join(dataPath,"Figures")
save_figures = True

current_data = 'Tm9_700_R_20231113.xlsx'

filePath =  os.path.join(dataPath,current_data)
_sheet_name = 'Relative_counts'

data_df = pd.read_excel(filePath, sheet_name=_sheet_name,index_col = 0)
number_of_columns = len(data_df)
pre_partners_ls = data_df.columns.tolist()
print(f'Total number of columns: {number_of_columns}')
print(f'All presynatic partners: \n {pre_partners_ls}')

# Applying a presence threshold based on NaN values
percetatge_prescence = 0.05
threshold = percetatge_prescence * len(data_df)

# Filter columns based on the threshold
filtered_data_df = data_df.dropna(thresh=threshold, axis=1)
filtered_pre_partners_ls = filtered_data_df.columns.tolist()
print(f'All presynatic partners after threshold: \n {filtered_pre_partners_ls}')

# Visualization of sorted inputs from hight to low
_data = filtered_data_df
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.5)
fig.suptitle(f'Wide field input partners (n = {number_of_columns}). \n {_sheet_name}')
sns.boxplot(data=_data[_data.mean().sort_values(ascending = False).index], ax=axs)
axs.set_ylabel('Relative Counts')
axs.set_xticklabels(axs.get_xticklabels(), rotation=90) 
plt.show()



#%% Anatomical RF analysis (area and column span)

#Defined variables for analysis purposes
min_desired_count = 3 # minimun desired number of contacts between pre and post neurons to be considered
single_column_diameter = 11.2 # in um (measured in FlyWire)
single_column_area = 100 # in um^2 (Assumed column as circle)
hemisphere = 'R'


########################################### Analysis of presynaptic neuron of interest ########################################### 

pre_neuron_type_ls = ['L3','Mi4','L4','Tm16','Dm12','C2','C3','Mi13','Tm1','Tm20','TmY10','TmY15','TmY17'] 


spatial_span_df_dict = {}
individual_spatial_span_df_dict = {}

for pre_neuron_type in pre_neuron_type_ls:
    print(f'Analyzing {pre_neuron_type}')

    ## Loading information  from excel files
    PC_disc = 'D'
    dataPath = f'{PC_disc}:\FlyWire-Data\database'
    date = '20230912'
    post_neuron_type = 'Tm9'
    fileName_post = f'{post_neuron_type} proofreadings.xlsx'
    filePath_post = os.path.join(dataPath,fileName_post)
    fileName_pre = f'{pre_neuron_type} proofreadings.xlsx'
    filePath_pre = os.path.join(dataPath,fileName_pre)

    #Loading file as DataFrame
    post_df = pd.read_excel(filePath_post)
    pre_df = pd.read_excel(filePath_pre)

    ## Filtering data
    # Selecting the R optic lobe IDs
    R_pre_df = pre_df[pre_df['hemisphere'] == hemisphere].copy()
    R_post_df = post_df[post_df['hemisphere'] == hemisphere].copy()
    # Selecting the backbone proofread IDs
    R_post_df = R_post_df[R_post_df['backbone proofread (Y/N)'] == 'Y'].copy()
    R_pre_df = R_pre_df[R_pre_df['backbone proofread (Y/N)'] == 'Y'].copy()
    # Selecting presynaptic cells ids
    pre_ids = R_pre_df['Updated_seg_id'].tolist()
    print(f'Number of neurons: {len( pre_ids)}')

    # Quick updates
    # Updating presynaptic neurons if they are not up-to-date
    if not np.unique(flywire.is_latest_root(pre_ids))[0]: #if not up-to-date
        print('Consider updating your ids in the original pre-neuron data set')
        pre_ids_update_df = flywire.update_ids(pre_ids, stop_layer=2, supervoxels=None, timestamp=None, dataset='production', progress=True)
        up_to_date_pre_ids = pre_ids_update_df['new_id'].tolist()
    else:
        up_to_date_pre_ids = [int(x) for x in pre_ids]
        print('All pre ids were up to date')


        ## Getting postsynaptic side ID
    post_ids = R_post_df['Updated_seg_id'].tolist()

    ## Updating postsynaptic neurons if they are not up-to-date
    if not np.unique(flywire.is_latest_root(post_ids))[0]: # if not up-to-date
        print('Consider updating your ids in the original post neuron data set')
        #Updating the IDs via Fafbseg
        post_ids_update_df = flywire.update_ids(post_ids, stop_layer=2, supervoxels=None, timestamp=None, dataset='production', progress=True)
        up_to_date_post_ids = post_ids_update_df['new_id']
    else:
        up_to_date_post_ids = [int(x) for x in post_ids]
        print('All post ids were up to date')


    ## Doing the pre to post match
    pre_post_counts, post_inputs = match_all_pre_to_single_post(up_to_date_post_ids, up_to_date_pre_ids, neuropile_mesh) 

    #############################################################################################
    ## Analysis f spatial span of preynaptic inputs to single, unicolumnar, postsynaptic cells ##

    # Synaptic counts filter
    pre_post_counts = pre_post_counts[pre_post_counts['pre_syn_count']>=min_desired_count].copy()

    ## Geeting information for all pre cells 
    pre_ls = pre_post_counts['pre_pt_root_id'].tolist() # all postsynaptic neurons

    # Fetch the inputs from presynaptic cells
    #TODO change "pre_inputs" to "pre_connectivity" (which includes inputs and outputs)
    pre_inputs = flywire.synapses.fetch_synapses(pre_ls, pre=True, post=True, attach=True, 
                                                 min_score=50, clean=True, transmitters=False, 
                                                 neuropils=True, batch_size=30, 
                                                 dataset='production', progress=True,mat= "live")

    # Filtering: keeping only synapses in the medulla
    pre_inputs = pre_inputs[pre_inputs['neuropil'] == neuropile_mesh].copy()
    len(pre_inputs)

    #Combining pre- and postsynpases XYZ values in single columns
    combine_xyz(pre_inputs) # Function that does the operation

    ## Quantificatino of the spatial span
    spatial_span_df, individual_spatial_span_df = calculate_spatial_span(up_to_date_post_ids, R_post_df, post_inputs, pre_post_counts, pre_inputs, single_column_area, single_column_diameter)
    spatial_span_df_dict[pre_neuron_type] = spatial_span_df
    individual_spatial_span_df_dict[pre_neuron_type] = individual_spatial_span_df
    

## Combining all dataframes and discarding outliers

# Initialize an empty list to store DataFrames
dfs_list = []

# Iterate through the dictionary
for neuron, df in individual_spatial_span_df_dict.items():
    #Discard outliers
    df = replace_outliers_with_nan(df, multiplier=1.5)
    # Add 'neuron' column to the DataFrame
    df['neuron'] = neuron
    if len(df) < 10:
        continue
    # Append the DataFrame to the list
    dfs_list.append(df)

# Concatenate DataFrames in the list
combined_individual_spatial_span_df = pd.concat(dfs_list, ignore_index=True)

print('Analysis done.')

###################################################### PLotting ##########################################################
##########################################################################################################################

####################################### Area and neurtransmitter (NT) identity ###########################################

# Define the neuron categories and their corresponding colors
glutamatergic_neurons = ['Mi13', 'Dm12']  # Choose blue
cholinergic_neurons = ['Tm1', 'Tm20', 'L3', 'L4', 'Mi4', 'TmY17', 'TmY10', 'Tm16']  # Choose green
gabaergic_neurons = ['C2', 'C3']  # Choose red

# Create the neuron_colors and legend_colors dictionaries
neuron_colors = {}
neuron_labels = {}
legend_colors = {}

# Assign colors to glutamatergic neurons and add labels
for neuron in glutamatergic_neurons:
    neuron_colors[neuron] = 'blue'
    neuron_labels[neuron] = 'glutamatergic'
    legend_colors['glutamatergic'] = 'blue'

# Assign colors to cholinergic neurons and add labels
for neuron in cholinergic_neurons:
    neuron_colors[neuron] = 'green'
    neuron_labels[neuron] = 'cholinergic'
    legend_colors['cholinergic'] = 'green'

# Assign colors to gabaergic neurons and add labels
for neuron in gabaergic_neurons:
    neuron_colors[neuron] = 'red'
    neuron_labels[neuron] = 'gabaergic'
    legend_colors['gabaergic'] = 'red'

# Select the desired columns and assign them to a new DataFrame
selected_columns = ['Area', 'neuron']
selected_df = combined_individual_spatial_span_df[selected_columns]

# Calculate the mean for each neuron category
mean_per_neuron = selected_df.groupby('neuron')['Area'].mean()

# Sort the neuron categories based on mean values in ascending order
sorted_neurons = mean_per_neuron.sort_values(ascending=True).index

# Create a list of colors and labels based on the neuron category of each data point
_palette = [neuron_colors[neuron] for neuron in sorted_neurons]
labels = [neuron_labels[neuron] for neuron in sorted_neurons]

# Create a figure handle
fig = plt.figure(figsize=(6, 2))  # Adjust the figure size as needed

# Plot the box plot with sorted order and custom palette
boxplot = sns.boxplot(x='neuron', y='Area', data=selected_df, palette=_palette, order=sorted_neurons)

# Remove left and top spines
boxplot.spines['right'].set_visible(False)
boxplot.spines['top'].set_visible(False)

# Rotate x-axis tick labels by 90 degrees
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90)

# Add labels and title
plt.xlabel('Neuron')
plt.ylabel('Area')
plt.title('Box Plot of Area by Neuron (Sorted by Mean in Ascending Order)')

# Add legend with labels and matching colors
legend_labels = list(set(labels))  # Get unique labels
legend_handles = [plt.Line2D([0], [0], marker='o', color=legend_colors[label], markerfacecolor=legend_colors[label], label=label) for label in legend_labels]
plt.legend(handles=legend_handles, labels=legend_labels, title='Neuron Type')

# Show the plot
plt.show()

if save_figures:    
    figure_title = 'All-neurons-spatial-span-NT.pdf'
    fig.savefig(fig_save_path + figure_title)
    print('FIGURE: All neurons spatial span saved')





################################# PLotting anatomical receptive field size for a single neuron #########################################

### Plotting histograms in same subplots:
_data = individual_spatial_span_df_dict['Dm12']
kde_only = True
filter_ouliers = False

# Create a single figure and axis for all subplots
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(40*cm, 12*cm))
fig.tight_layout(pad=3)

# Define colors 
color_rgb = (226 / 255, 26 / 255, 28 / 255)  # Red in RGB 
       

# Outlier filter
if filter_ouliers:
    _data = replace_outliers_with_nan(_data, multiplier=1.5)

# Plot histograms and/or KDE in the respective subplots
histograms = [_data['Column_span_projected'], _data['Area'], _data['Num_columns'], _data['Diameter_projected']]
subplot_labels = ['Column span projected', 'Presynaptic cells area (um^2)', 'Columns in area', 'Diameter projected']
histograms_bin_width = [None, None, None, None]

# Reset the index to avoid reindexing issues
for i, data in enumerate(histograms):
    data = data.reset_index(drop=True)

    if kde_only:
        sns.kdeplot(data, ax=axs[i], label=f'{pre_neuron_type}- {len(data)}', color=color_rgb)
        axs[i].set_ylabel('Density')
    else:
        sns.histplot(data, binwidth=histograms_bin_width[i], ax=axs[i], label=f'{pre_neuron_type} - {len(data)}', color=color_rgb, alpha=0.075, 
                         kde=True, element="step", fill=True, stat="count")
        axs[i].set_ylabel('Count')

    axs[i].set_xlabel(subplot_labels[i])
        
    # Add mean and median lines
    add_mean_median_lines(data, axs[i], 'red', 'k')
        
    #Remove spines
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

# Last subplot
_data_reset_index = _data.reset_index()
sns.histplot(
_data_reset_index, x='Column_span_projected', y='Num_columns',
bins=30, discrete=(True, True), log_scale=(False, False), color=color_rgb,
cbar=True, cbar_kws=dict(shrink=.75), ax=axs[4], label=f'{pre_neuron_type} - {len(data)}')
axs[4].set_xlabel('Column span')
axs[4].set_ylabel('Columns in area')

    
# Add mean and median lines for x- and y-axis
add_mean_median_lines(_data_reset_index['Column_span_projected'], axs[4], 'red', 'k')
add_mean_median_lines(_data_reset_index['Num_columns'], axs[4], 'red', 'k', vertical=False)

# Add legends to the subplots
for i in range(5):
    axs[i].legend()

# Show the figure
plt.show()

if save_figures:    
    figure_title = f'\{pre_neuron_type}-neuron-spatial-span-quantification.pdf'
    fig.savefig(fig_save_path+figure_title)
    print('FIGURE: Spatial span saved')


print('Plotting done.')