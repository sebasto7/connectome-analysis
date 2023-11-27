# -*- coding: utf-8 -*-
"""
Created on Mon June 12 14:43:16 2023

@author: smolina

variability analysis of presynaptic inputs
"""
#%% 
#Importing packages
import navis
import fafbseg
from fafbseg import flywire
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from itertools import combinations
from openpyxl import load_workbook
from mpl_toolkits.mplot3d import Axes3D
import os
import math
import random
from caveclient import CAVEclient
client = CAVEclient('flywire_fafb_production')

#Importing custom functions from helper file
from helper import filter_values, create_column_c, add_n_labels, replace_outliers_with_nan, permutation_test, calculate_correlation_and_p_values, cosine_similarity_and_clustering

#%% 
############################################# PLOTS GENERAL SETTINGS ##########################################
###############################################################################################################
#General style
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}
axes = {'labelsize': 8, 'titlesize': 8}
ticks = {'labelsize': 4}
legend = {'fontsize': 8}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', **ticks)
plt.rc('ytick', **ticks)

#Saving text for pdfs in a proper way
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cm = 1/2.54  # centimeters in inches

#Sorting options for plots
sort_by = 'mean_abs_count' # 'median_rank', 'median_abs_count', 'median_rel_count', 'column_%', 'mean_abs_count' (used in the Nature submission)
#Choosing type of data for cosine similarity and cluster analysis
relative_or_absolute = 'relative-counts' # 'absolute-counts', 'relative-counts' 

#Plots by category (e.g. dorsal (D) vs ventral (V) or rigth (R) vs left (L))
category_column = 'dorso-ventral'# 'dorso-ventral', 'hemisphere'
cosine_subgroups = ['D', 'V'] # ['D', 'V'], ['R', 'L']

#Colors
color_cat_set = "Set1" # select a set for a seaborn color palette
hex_color = 'light:#458A7C' # Tm9: 'light:#458A7C', Tm1: 'light:#D57DA3', Tm2: 'light:#a6761d'
neuron_color = '#458A7C' # Tm9: '#458A7C', Tm1: '#D57DA3', Tm2: '#a6761d'
#YES-NO options
plot_data = True
save_figures = True
exclude_outliers = True # Plot variability without outliers
stacked_plots = False 


#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################
#General analysis information (user defined)

#Count coverage
desired_coverage = 80 # in percent
plus_minus = 5 # in percent

#Synaptic counts
min_desired_count = 3 # minimun number of synapses to consider in the analysis # Tm1:4, Tm9:3, Tm2:4, Tm4:3
presence_threshold = 0.05 # If a partner is in less than 5% of the columns, it will be discarted for further visualizations

#Neuron counts
desired_quantile = 0.8 # selected quantile for neuron counts (currently not in use)

#Main presynaptic partner
last_input_neuron = 1000 # last input to be considered in summary dataframes across columns

#Dorsal-ventral filter
d_v_filter = False # 
selection_area = '' # 'D', 'V'

#Analysis of heterogeneous partners only:
discard_homogeneous = False   # To analyse all partnes, make it False
thr_homogenous = 0.9 # Threshold about which a partner is defined as homogenous (e.g present in 90% or 75% of the columns)

#Clustering options
cluster_with_dendrogram = True

#For permutation test in pearson correlation
permutation_analysis = False
num_permutations = 1000
_seed = 42 # For the random selection of Tm9-columns (rows in data frames) from the complete dataset

#Data set 
optic_lobe = 'R'  # 'L', 'R', 'L_R'
dataset_name = f'FAFB_{optic_lobe}_{selection_area}'
mesh_ME = 'ME_L' # 'ME_R' , 'ME_L' 
mesh_LO = 'LO_L' # 'LO_R' , 'LO_L'
mesh_azim = 16# -18 for ME_R, 16 for ME_L
mesh_elev = -50 # -148 for ME_R, -50 for ME_L
neuron_of_interest = 'Tm9' 
check_input_neuron = 'L3'
check_input_min_weight = 30
instance_id_column = 'optic_lobe_id' # 'optic_lobe_id', 'column_id'

##
mesh_OL_L = 'ME_R'
mesh_OL_R = 'ME_L'
mesh_azim_L = -18# -18 for ME_R, 16 for ME_L
mesh_elev_L = -148 # -148 for ME_R, -50 for ME_L
mesh_azim_R = 16# -18 for ME_R, 16 for ME_L
mesh_elev_R = -50 # -148 for ME_R, -50 for ME_L

# Cluster information
analyzing_cluster = False
cluster_id = 'C1'
save_clusters_txt = True

#Path and file
PC_disc = 'D'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Excels\drive-data-sets'# '~\Connectomics-Data\FlyWire\Excels\drive-data-sets' ,'~\Connectomics-Data\FlyWire\Excels\drive-data-sets\submission_nature',
fileDate = '700_20231113' # 20230823, 490_20231109, 700_20231111
fileName = f'{neuron_of_interest}_neurons_input_count_{optic_lobe}_{fileDate}.xlsx'
#fileName = f'Tm9_neurons_input_count_L_R_OA_subtypes_20230718.xlsx' # Remove this line after OA plots are done
fileName_database = f'{neuron_of_interest} proofreadings_{fileDate}.xlsx'
fileName_NT = 'NT_identity_optic_lobe_neurons.xlsx'

#Data set subselection
subselection_filter = True
#Manual selection
subselection_id_columns = [] # list of optic_lobe_ids
#Loading file for subselection
subselection_file = True
txtPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Txts\optic_lobes_ids'#r'C:\Connectomics-Data\FlyWire\Txts\optic_lobes_ids'
fileName_txt = f'Tm9_700_healthy_L3_over_30_{optic_lobe}.txt' # Tm9_300_healthy_L3_{optic_lobe}.txt, Tm9_490_healthy_L3_over_10_{optic_lobe}.txt, 'Tm9_490_{optic_lobe}.txt', 'Tm9_700_{optic_lobe}.txt',  'Tm2_healthy_L3_{optic_lobe}.txt', 'Tm9_healthy_L3_{optic_lobe}.txt', 'Tm9_D_patch_L.txt' 'Tm9_cosine_similarity_C2_{optic_lobe}.txt', 'Tm9_sparse_healthy_R.txt', 'Tm9_sparse_L.txt' , 'Tm9_dark_L3_R.txt', 'Tm9_sparse_healthy_L3_L_R.txt', 'Tm9_consine_similarity_cluster_1_2_R.txt'


# Healthy columns based on lamina detachement and damage L3s
keep_only_healthy_columns = False # Only good if your data subselection does not considere it already (not being used)


#Expansion microscopy
ExM_dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Excels\expansion-microscopy'# r'C:\Connectomics-Data\FlyWire\Excels\expansion-microscopy'


#Processed data saving path
saving_processed_data = True
output_dataPath = f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data'#r'C:\Connectomics-Data\FlyWire\Processed-data'


#%%  
############################################# ADVANCE CLUSTER OPTIONS #######################################

# When running data just for a cluster, making sure this option are like this:
if analyzing_cluster:
    print(f'\nAnalyzing cluster: {cluster_id}')
    subselection_filter = True
    fileName_txt = f'Tm9_cosine_similarity_{cluster_id}_{optic_lobe}.txt'
    cluster_with_dendrogram = False
    discard_homogeneous = False
    permutation_analysis = True
    if optic_lobe == 'R':
        user_defined_sorted_column_order = ['L3','CT1','Mi4','Tm16','L4','Dm12','C3',
                                            'Tm20','Tm1','putative-fru-N.I.','C2','TmY17',
                                            'PS125','ML1','OA-AL2b2','TmY15','Me-Lo-2-N.I.','Tm2','Mi13']

    elif optic_lobe == 'L':
        user_defined_sorted_column_order = ['L3','CT1','Mi4','Tm16','putative-fru-N.I.',
                                            'L4','PS125','Dm12','Tm20','C3','Tm1','ML1',
                                            'Tm5c','Mi13','Tm2','OA-AL2b2','C2','TmY17'] 

    #For Tm9::R, ['L3','CT1','Mi4','Tm16','L4','Dm12','C3','Tm20','Tm1','putative-fru-N.I.','C2','TmY17','PS125','ML1','OA-AL2b2','TmY15','Me-Lo-2-N.I.','Tm2','Mi13']
    #For Tm9::L, ['L3','CT1','Mi4','Tm16','putative-fru-N.I.','L4','PS125','Dm12','Tm20','C3','Tm1','ML1','Tm5c','Mi13','Tm2','OA-AL2b2','C2','TmY17']

    #Loading the complete (unclustered) data set
    file_name = f'{neuron_of_interest}_{dataset_name}.xlsx'
    processed_dataPath = os.path.join(output_dataPath,file_name)
    dataset_abs_df = pd.read_excel(processed_dataPath, sheet_name='Absolut_counts', index_col = 0)
    dataset_abs_df = dataset_abs_df[user_defined_sorted_column_order].copy()
    dataset_rel_df = pd.read_excel(processed_dataPath, sheet_name='Relative_counts', index_col = 0)
    dataset_rel_df = dataset_rel_df[user_defined_sorted_column_order].copy()
    

#%% 
################################################## PRE-ANALYSIS ###############################################
############################################# ELECTRON MICROSCOPY (EM) ########################################

#Loading FAFB data from our data base (excel file)
filePath = os.path.join(dataPath,fileName)
df = pd.read_excel(filePath)

#Loading FAFB data from our data base (excel file)
filePath = os.path.join(dataPath,fileName_database)
database_df = pd.read_excel(filePath)

#Loading neurotransmitter (NT) identity (excel file)
filePath = os.path.join(dataPath,fileName_NT)
NT_df = pd.read_excel(filePath)

#Loading subselection file
if subselection_file:
    filePath = os.path.join(txtPath,fileName_txt)
    txt_data = pd.read_csv(filePath, sep=" ", header=None)
    subselection_id_columns = list(set(txt_data[0].tolist()))


#Dropping rows realted to manual proofreading and strings
#('asdf' was added as a walk-around to set that column values as type str)
if df["postsynaptic_ID"][0] == 'asdf': 
    df = df.iloc[1: , :]
    df.reset_index(inplace=True,drop=True)

if database_df["seg_id"][0] == 'asdf': 
    database_df = database_df.iloc[1: , :]
    database_df.reset_index(inplace=True,drop=True)
    
# Removing unnecessary rows ment for human use only.
# Getting rid of info rows with no connectomics data
if 'INPUTS PROOFREAD' in df.values: 
    df = df[df['presynaptic_ID']!= 'INPUTS PROOFREAD'].copy() 

#Adjusting column names to meet the naming of the FIB25 data sets in NeuPrint (interesting for future use or comparison)
#Creating new columns
df['instance_pre'] = df['symbol'] + '::' + df[instance_id_column] + ':' + df['dorso-ventral'] 
df['instance_post'] = neuron_of_interest + '::' + df[instance_id_column] + ':' + df['dorso-ventral'] 
df['type_post'] = neuron_of_interest
df['counts']= df['counts'].astype(int)

#Sorting rows based on count synapse number
df = df.groupby(['instance_post']).apply(lambda x: x.sort_values(['counts'], ascending = False)).reset_index(drop=True)
#Ranking the neurons
df['rank'] = df.groupby(['instance_post']).cumcount().tolist()
#Renaming columns
df.rename(columns={'presynaptic_ID':'bodyId_pre', 'counts':'W','Updated_counts':'W_new', 'postsynaptic_ID':'bodyId_post','symbol':'type_pre'}, inplace = True)
#Keeping only columns of interest
cols_to_keep = ['rank','patch_id','column_id','optic_lobe_id','detached_lamina (Y/N)','healthy_L3 (Y/N)','hemisphere','dorso-ventral','instance_pre','type_pre','bodyId_pre','instance_post','type_post','bodyId_post','W', 'W_new']
df = df[cols_to_keep].copy()

#Filtering out faulty columns
if keep_only_healthy_columns:
    df = df[df['detached_lamina (Y/N)'] == 'N'].copy() #Keep only the columns below a healthy lamina
    df = df[df['healthy_L3 (Y/N)'] == 'Y'].copy() #Keep only the columns with healthy (non-dark) L3

#Subselection filter:
if subselection_filter:
    df = df[df['optic_lobe_id'].isin(subselection_id_columns)].copy()
    if d_v_filter:
        df = df[df['dorso-ventral'] == selection_area].copy()

# Keeping a dataframe without any synaptic ot N.I filter
df_0 = df.copy()

#Calculating relative counts (weigths)
df['column_percent'] = round((df['W_new'] / df.groupby('instance_post')['W_new'].transform('sum'))*100,2)
df['cumulative_column_percent'] = df.groupby('instance_post')['column_percent'].cumsum()
 
#TODO: replase this approach of df[df['W_new']== min_desired_count] to look for total coverage.
#Initial coverage (mean-+std  % of all inputs across columns) before synaptic-count filter
desired_count_df = df[df['W_new']== 1].copy() # 1 as the lowest number for a contact
last_percent_with_desired_count = pd.DataFrame(desired_count_df.groupby(['instance_post'])['cumulative_column_percent'].max()) #Covarage across columns per the desired_count
coverage = f"{round(last_percent_with_desired_count['cumulative_column_percent'].mean(),2)}+-{round(last_percent_with_desired_count['cumulative_column_percent'].std(),2)}"

#Initial coverage (mean-+std  % of all inputs across columns) after synaptic-count filter
desired_count_df = df[df['W_new']== min_desired_count].copy()
last_percent_with_desired_count = pd.DataFrame(desired_count_df.groupby(['instance_post'])['cumulative_column_percent'].max()) #Covarage across columns per the desired_count
initial_coverage = f"{round(last_percent_with_desired_count['cumulative_column_percent'].mean(),2)}+-{round(last_percent_with_desired_count['cumulative_column_percent'].std(),2)}"

#Filtering out inidentfied segmetns (important: after the calculation of relative counts) 
if 'N.I.' in df.values: # Removing non identified (N.I.) inputs
    df = df[df['type_pre']!= 'N.I.'].copy() # Getting rid of info rows with no connectomics data

if 'chunk-N.I.' in df.values: # Removing non identified (N.I.) inputs
    df = df[df['type_pre']!= 'chunk-N.I.'].copy() # Getting rid of info rows with no connectomics data

#Final coverage (mean-+std % of all inputs across columns) after N.I-partners filter
desired_count_df = df[df['W_new']== min_desired_count].copy()
last_percent_with_desired_count = pd.DataFrame(desired_count_df.groupby(['instance_post'])['cumulative_column_percent'].max()) #Coverage across columns per the desired_count
final_coverage = f"{round(last_percent_with_desired_count['cumulative_column_percent'].mean(),2)}+-{round(last_percent_with_desired_count['cumulative_column_percent'].std(),2)}"

#Ranking the neurons again after previous filtering
df['rank'] = df.groupby(['instance_post']).cumcount().tolist()

#Getting the neuron´s meshes
root_ids = df['bodyId_post'].unique().tolist()
#m_all = flywire.get_mesh_neuron(root_ids) # This is not yet needed. Use for plotting neuron`s anatomy
#print('- Got all neuron meshes -')

#Looking at number of neurons and cell types per column
id_column = df[instance_id_column].unique().tolist()
cell_counts_df = pd.DataFrame(df.groupby(['instance_post']).agg({'type_pre':'count', 'instance_pre':'count'}))


#############################################   PRINT STATEMENTS  ###########################################
#Printing useful information
print(f'\nThe following column ids (n={len(id_column)}) are part of the analysis: \n {id_column}')
print(f'\nInitial coverage (%) for syn >= 1: {coverage}')
print(f'\nInitial coverage (%) for syn >= {min_desired_count}: {initial_coverage}')
print(f'\nFinal coverage (%) after dropping N.I. for syn >= {min_desired_count}: {final_coverage}\n') 
print(last_percent_with_desired_count)
columns_wiht_X_neuron_iput = df[(df['type_pre'] == check_input_neuron) & (df['W_new'] > check_input_min_weight)]['optic_lobe_id'].unique().tolist()
print(f"Columns with L3 as input neuron (syn > {check_input_min_weight}): {len(columns_wiht_X_neuron_iput)} out of {len(id_column)} \n {columns_wiht_X_neuron_iput}")

## For modifying an excel file in a particular way 
# with open('D:\Connectomics-Data\FlyWire\Txts\optic_lobes_ids\Tm9_700_healthy_L3_over_30_R.txt', 'r') as file:
#     content = file.read()

# # Replace the characters as needed
# content = content.replace("', '", "\n").replace("'", "")

# # Save the modified content to a new file
# with open('D:\Connectomics-Data\FlyWire\Txts\optic_lobes_ids\Tm9_700_healthy_L3_over_30_R_updated.txt', 'w') as new_file:
#     new_file.write(content)



#%% 
################################################## PRE-ANALYSIS ###############################################
############################################ EXPANSION MICROSCOPY (ExM) ########################################

#Loading files and transforming data

ExM_absolut_counts = {}
for file in os.listdir(ExM_dataPath):
    filePath = os.path.join(ExM_dataPath,file)
    curr_df = pd.read_csv(filePath,dtype=str)
    curr_df.drop('Unnamed: 0', axis=1, inplace=True)
    curr_df['type_post'] = neuron_of_interest
    curr_df.rename(columns={'counts':'W_new','symbol':'type_pre'}, inplace = True)
    curr_df['instance_post'] = neuron_of_interest + '::' + curr_df['column_id']
    curr_df['fly_ID'] = curr_df['fly_ID'] + '-ExM'
    # Convert 'W' column to integer
    curr_df['W_new'] = curr_df['W_new'].astype(int)
    curr_df = curr_df[curr_df['W_new'] >= min_desired_count].copy()
    curr_df.reset_index(inplace=True)
    ExM_absolut_counts[curr_df['type_pre'][0]] = curr_df


#%% 
################################################## ANALYSIS ###################################################
###############################################################################################################
###############################################################################################################


# Additional useful data frames. 

############################################# IDENTITY AND CONNECTIONS ########################################
###############################################################################################################

identity_dict = {} # instance of input neurons
identity_type_dict = {}
identity_type_middle_rank_dict = {}
abs_connections_dict = {} # absolut number of connections for tables
rel_connections_dict = {} # relative number of connections for tables
index_name_ls = [] # For scatterplots
input_rank_ls = [] # For scatterplots
abs_connection_ls = [] # For scatterplots
rel_connection_ls = [] # For scatterplots

# Summary table of the main inputs across columns
identity_df = pd.DataFrame()
abs_connections = pd.DataFrame()
rel_connections = pd.DataFrame()
for instance in df['instance_post'].unique():
    curr_df = df[df['instance_post'] ==instance].copy()
    curr_df.reset_index(inplace = True) # Important step

    # Adding the perceptatge of inputs
    N_sum = curr_df['W_new'].sum()
    N_percentatge = curr_df['W_new'].tolist()/N_sum * 100
    curr_df['W_percentatge'] = N_percentatge.round(2) #rounding to the second decimal place

    #Synaptic strengh filter
    curr_df = curr_df[curr_df['W_new']>=min_desired_count].copy()

    #For table across columns
    identity_dict[curr_df['instance_post'].unique().tolist()[0]] = curr_df['instance_pre'][0:last_input_neuron] # curr_df['instance_post'].unique().tolist()[0] expects a single value
    identity_df= pd.DataFrame(identity_dict) # Here it concatenates at every loop
    identity_type_dict[curr_df['instance_post'].unique().tolist()[0]] = curr_df['type_pre'][0:last_input_neuron]
    identity_type_df= pd.DataFrame(identity_type_dict) # Here it concatenates at every loop
    identity_type_middle_rank_dict[curr_df['instance_post'].unique().tolist()[0]] = curr_df['type_pre'][last_input_neuron:]
    identity_type_middle_rank_df= pd.DataFrame(identity_type_middle_rank_dict) # Here it concatenates at every loop
    
    
    #print(f"Input coverage up to the {_end}th input: {round(curr_df['W_percentatge'][0:7].sum(),2)} %")
    abs_connections_dict[curr_df['instance_post'].unique().tolist()[0]] = curr_df['W_new'][0:last_input_neuron]
    rel_connections_dict[curr_df['instance_post'].unique().tolist()[0]] = curr_df['W_percentatge'][0:last_input_neuron]
    abs_connections_df= pd.DataFrame(abs_connections_dict) # Here it concatenates at every loop
    rel_connections_df= pd.DataFrame(rel_connections_dict) # Here it concatenates at every loop
    
    #For scatter plots
    index_name_ls = index_name_ls + ([instance] * len(curr_df['W_new'][0:last_input_neuron]))
    input_rank_ls = input_rank_ls + list(range(0,len(curr_df['W_new'][0:last_input_neuron]))) #Concatenating lists across loops
    abs_connection_ls = abs_connection_ls + curr_df['W_new'][0:last_input_neuron].tolist() #Concatenating lists across loops
    rel_connection_ls = rel_connection_ls + curr_df['W_percentatge'][0:last_input_neuron].tolist() #Concatenating lists across loops
    
#Adding total sums information to some dataframes
rel_connections_df.loc['Total',:] = rel_connections_df.sum(axis=0).tolist()
rel_connections_df
abs_connections_df.loc['Total',:] = abs_connections_df.sum(axis=0).tolist()
abs_connections_df
    

################################################# RANKS #######################################################
###############################################################################################################
#Analysis across ranks
rank_rel_abs_df = pd.DataFrame(index=index_name_ls)
rank_rel_abs_df ['Abs_connection'] = abs_connection_ls
rank_rel_abs_df ['Rel_connection'] = rel_connection_ls
rank_rel_abs_df ['Connection_rank'] = input_rank_ls

mean_abs_ls = [] # For variability analysis 
mean_rel_ls = [] # For variability analysis 
std_abs_ls = [] # For variability analysis 
std_rel_ls = [] # For variability analysis 
CV_abs_ls = [] # For variability analysis (CV = coefficient of variation)
CV_rel_ls = [] # For variability analysis (CV = coefficient of variation)
p50_abs_ls = [] # For variability analysis 
p50_rel_ls = [] # For variability analysis 

for rank in rank_rel_abs_df['Connection_rank'].unique():
    
    curr_df = rank_rel_abs_df[rank_rel_abs_df['Connection_rank'] == rank].copy()
    #Variability indexes
    mean_abs_ls.append(round(np.mean(curr_df['Abs_connection'].tolist()),2)) 
    mean_rel_ls.append(round(np.mean(curr_df['Rel_connection'].tolist()),2)) 
    std_abs_ls.append(round(np.std(curr_df['Abs_connection'].tolist()),2)) 
    std_rel_ls.append(round(np.std(curr_df['Rel_connection'].tolist()),2))
    CV_abs_ls.append(round(np.std(curr_df['Abs_connection'].tolist())/np.mean(curr_df['Abs_connection'].tolist()),2)) 
    CV_rel_ls.append(round(np.std(curr_df['Rel_connection'].tolist())/np.mean(curr_df['Rel_connection'].tolist()),2))
    p50_abs_ls.append(round(np.percentile(curr_df['Abs_connection'].tolist(),50),2)) 
    p50_rel_ls.append(round(np.percentile(curr_df['Rel_connection'].tolist(),50),2))
    
stats_ranked_df = pd.DataFrame(index=rank_rel_abs_df['Connection_rank'].unique())
stats_ranked_df ['Mean_abs'] = mean_abs_ls
stats_ranked_df ['Mean_rel'] = mean_rel_ls
stats_ranked_df ['Std_abs'] = std_abs_ls
stats_ranked_df ['Std_rel'] = std_rel_ls
stats_ranked_df ['CV_abs'] = CV_abs_ls
stats_ranked_df ['CV_rel'] = CV_rel_ls
stats_ranked_df ['P50_abs'] = p50_abs_ls
stats_ranked_df ['P50_rel'] = p50_rel_ls

# General rank sorting
#First axis: synaptic filtered heatmap, top-rank data
top_rank_df = df[(df['W_new']>=min_desired_count) & (df['rank']<last_input_neuron)].copy()
curr_df = top_rank_df[['rank', 'type_pre', 'instance_post' ]].copy()
curr_df.set_index('instance_post', inplace = True)
# Choosing the rank for 'type_pre' as the rank of the first ocurrance
rank_df = curr_df.pivot_table(values='rank', index=curr_df.index, columns='type_pre', aggfunc='first').copy()

#Adding the median of the existing values in other row if NaN (soft penalization)
median_values = rank_df.median()  # Calculate the median for each column4
abscence_penalty = 100
#rank_df.fillna(abscence_penalty, inplace = True) # In case we wanna add a penalty value in columns where the cell type is absent. 
if sort_by == 'median_rank':
    rank_column_order = rank_df.median().sort_values().index.tolist()
    sorted_column_order = rank_df.median().sort_values().index.tolist()


############################################# INSTANCES OF NEURONS ############################################
###############################################################################################################
### Counting instances

counting_instances_df = pd.DataFrame()
for c in identity_df.columns:
    temp_df = pd.DataFrame(identity_df[c].value_counts()) # counting instances
    ##TODO change the column name here from Tm9-A1 to A1, for example
    temp_df.index = list(map(lambda x: x[0:x.find('::')],temp_df.index))# transforming from instance to type
    counting_instances_df = pd.concat([counting_instances_df, temp_df], axis=1) # Concatenating columns
    
    
### Counting NaNs (counting the absence of a neuron across columns) 

abs_NaNs_count_ls = []
abs_value_count_ls = []
rel_NaNs_count_ls = []
rel_value_count_ls = []
for neuron in counting_instances_df.index:
    temp_NaN_count = counting_instances_df.loc[neuron].isnull().sum()
    temp_column_number = len(counting_instances_df.loc[neuron])
    abs_NaNs_count_ls.append(temp_NaN_count)
    abs_value_count_ls.append(temp_column_number - temp_NaN_count)
    rel_NaNs_count_ls.append(temp_NaN_count/temp_column_number)
    rel_value_count_ls.append((temp_column_number - temp_NaN_count)/temp_column_number)
    
abs_presence_absence_df= pd.DataFrame({'Presynaptic neuron': counting_instances_df.index.tolist(),
                                   'Present': abs_value_count_ls,
                                  'Absent': abs_NaNs_count_ls})

rel_presence_absence_df= pd.DataFrame({'Presynaptic neuron': counting_instances_df.index.tolist(),
                                   'Present': rel_value_count_ls,
                                  'Absent': rel_NaNs_count_ls})

thr_rel_presence_absence_df = rel_presence_absence_df[rel_presence_absence_df['Present']> presence_threshold].copy()

#Sorting dataframe
sorted_abs_presence_absence_df = abs_presence_absence_df.sort_values(by=['Present'], ascending=False)
sorted_rel_presence_absence_df = rel_presence_absence_df.sort_values(by=['Present'], ascending=False)
sorted_thr_rel_presence_absence_df = thr_rel_presence_absence_df.sort_values(by=['Present'], ascending=False)

if sort_by == 'column_%':
    sorted_column_order = sorted_thr_rel_presence_absence_df['Presynaptic neuron'].tolist()


#Neuron filter based on "presence_threshold"
presence_threshold_neuron_filter = thr_rel_presence_absence_df['Presynaptic neuron'].tolist()

#Filters for homogeneous and heterogenoue partners
#An homogeneous partner is beind define as that one present in at least X% of the columns (e.g. 75%)
if discard_homogeneous == True:
    presence_threshold_neuron_filter = thr_rel_presence_absence_df[ thr_rel_presence_absence_df['Present']< thr_homogenous]['Presynaptic neuron'].tolist()
    excluded_partners = thr_rel_presence_absence_df[ thr_rel_presence_absence_df['Present']> thr_homogenous]['Presynaptic neuron'].tolist()
    print(f'\nPartners being INcluded in the analyis:\n {presence_threshold_neuron_filter}')
    print(f"\nPartners being EXcluded from the analyis:\n {excluded_partners}")


########################################## PRESENCE - ABSENCE of a partner ####################################
###############################################################################################################
# Turning the dataset to binary
binary_df = counting_instances_df.T.copy()
binary_df[binary_df.notnull()] = 1
#binary_df[binary_df.isnull()] = 0 # Fo now, do not excecuto so that values remain NaN

#sorting
sum_column_order = binary_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
binary_sum_sorted_df = binary_df[sum_column_order] # swapping order of columns




##########################################  RELATIVE SYNAPTIC COUNTS ##########################################
###############################################################################################################
#For REALTIVE QUANTIFICATIONS (using 'column_percent')

#Checking the most popular presynaptic partners based on: 

#1) neuron counts across columns
top_rank_popular_neuron_ls = identity_type_df.stack().value_counts().index.tolist()
top_rank_popular_neuron_ls

#include nan neurons
top_rank_nan = top_rank_popular_neuron_ls.copy()
top_rank_nan.append('N.I')

#2) total percentatge of synaptic count across columns using top-rank neuron data or all data above syn threshold! 
#Synaptic strengh filter
#All data above syn threshold
syn_df = df[df['W_new']>=min_desired_count].copy()
syn_type_df = pd.DataFrame(syn_df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe, filtered

#include NaNs
df_nans = df.copy()
df_nans['type_pre'] = df_nans['type_pre'].fillna('N.I.')
nan_type_df = df_nans.groupby(['instance_post', 'type_pre']).agg({'W_new':sum ,'column_percent':sum})

#Top-rank neuron data
top_rank_df = df[(df['W_new']>=min_desired_count) & (df['rank']<last_input_neuron)].copy()
top_rank_type_df = pd.DataFrame(top_rank_df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe, filtered
type_df = pd.DataFrame(df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe

popularity_rel_connections_dict = {}
syn_popularity_rel_connections_dict = {}
top_rank_popularity_rel_connections_dict = {}
for pre in top_rank_popular_neuron_ls: # popular neurons
    
    #Synaptic filter included
    #Top-rank neuron data
    temp_percent_ls = []
    for post in top_rank_type_df.index.levels[0].tolist(): #Columns
        if pre in top_rank_type_df.loc[post].index:
            temp_percent_ls.append(round(top_rank_type_df.loc[post,pre]['column_percent'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    top_rank_popularity_rel_connections_dict[pre] = temp_percent_ls
    
    #Synaptic strengh filter
    temp_percent_ls = []
    for post in syn_type_df.index.levels[0].tolist(): #Columns
        if pre in syn_type_df.loc[post].index:
            temp_percent_ls.append(round(syn_type_df.loc[post,pre]['column_percent'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    syn_popularity_rel_connections_dict[pre] = temp_percent_ls
    
    #No filter
    temp_percent_ls = []
    for post in type_df.index.levels[0].tolist(): #Columns
        if pre in type_df.loc[post].index:
            temp_percent_ls.append(round(type_df.loc[post,pre]['column_percent'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    popularity_rel_connections_dict[pre] = temp_percent_ls
        
top_rank_popularity_rel_df = pd.DataFrame(top_rank_popularity_rel_connections_dict)
top_rank_popularity_rel_df.index = top_rank_type_df.index.levels[0]

syn_popularity_rel_df = pd.DataFrame(syn_popularity_rel_connections_dict)
syn_popularity_rel_df.index = syn_type_df.index.levels[0]

popularity_rel_df = pd.DataFrame(popularity_rel_connections_dict)
popularity_rel_df.index = type_df.index.levels[0]


#with nans
nan_popularity_rel_connections_dict = {}
for pre in top_rank_nan:
    temp_percent_ls = []
    for post in nan_type_df.index.levels[0].tolist(): #Columns
        if pre in nan_type_df.loc[post].index:
            temp_percent_ls.append(round(nan_type_df.loc[post,pre]['column_percent'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    nan_popularity_rel_connections_dict[pre] = temp_percent_ls

nan_popularity_rel_df = pd.DataFrame(nan_popularity_rel_connections_dict)
nan_popularity_rel_df.index = nan_type_df.index.levels[0]

#Taking the most popular of the popular based on descending values of the mean synaptic counts acroos colums
popularity_neuron_based_on_count_percentatge_ls = syn_popularity_rel_df.aggregate('mean', axis = 0).sort_values(ascending =False).index.tolist()
print('\n All partners:')
print(popularity_neuron_based_on_count_percentatge_ls[0:last_input_neuron])


if sort_by == 'median_abs_count':
    syn_popularity_rel_df_zeros = syn_popularity_rel_df.fillna(0) # Replacing NaNs with zeros
    sorted_column_order = syn_popularity_rel_df.median().sort_values(ascending=False).index.tolist()

if sort_by == 'mean_abs_count':
    syn_popularity_rel_df_zeros = syn_popularity_rel_df.fillna(0) # Replacing NaNs with zeros
    sorted_column_order = syn_popularity_rel_df_zeros.mean().sort_values(ascending=False).index.tolist()



##########################################  ABSOLUTE SYNAPTIC COUNTS ##########################################
###############################################################################################################


#For ABSOLUTE QUANTIFICATIONS (using W')
#Checking the most popular presynaptic partners based on: 

#1) neuron counts across columns
top_rank_popular_neuron_ls = identity_type_df.stack().value_counts().index.tolist()
top_rank_popular_neuron_ls

#2) total percentatge of synaptic count across columns using top-rank neuron data or all data above syn threshold! 
#Synaptic strengh filter
#All data above syn threshold
syn_df = df[df['W_new']>=min_desired_count].copy()
syn_type_df = pd.DataFrame(syn_df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe, filtered

#Top-rank neuron data
top_rank_df = df[(df['W_new']>=min_desired_count) & (df['rank']<last_input_neuron)].copy()
top_rank_type_df = pd.DataFrame(top_rank_df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe, filtered
type_df = pd.DataFrame(df.groupby(['instance_post', 'type_pre']).agg({'W_new':sum, 'column_percent':sum})) #Neuron type dataframe

popularity_abs_connections_dict = {}
syn_popularity_abs_connections_dict = {}
top_rank_popularity_abs_connections_dict = {}
for pre in top_rank_popular_neuron_ls: # popular neurons
    
    #Synaptic filter included
    #Top-rank neuron data
    temp_percent_ls = []
    for post in top_rank_type_df.index.levels[0].tolist(): #Columns
        if pre in top_rank_type_df.loc[post].index:
            temp_percent_ls.append(round(top_rank_type_df.loc[post,pre]['W_new'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    top_rank_popularity_abs_connections_dict[pre] = temp_percent_ls
    
    #Synaptic strengh filter
    temp_percent_ls = []
    for post in syn_type_df.index.levels[0].tolist(): #Columns
        if pre in syn_type_df.loc[post].index:
            temp_percent_ls.append(round(syn_type_df.loc[post,pre]['W_new'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    syn_popularity_abs_connections_dict[pre] = temp_percent_ls
    
    #No filter
    temp_percent_ls = []
    for post in type_df.index.levels[0].tolist(): #Columns
        if pre in type_df.loc[post].index:
            temp_percent_ls.append(round(type_df.loc[post,pre]['W_new'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    popularity_abs_connections_dict[pre] = temp_percent_ls
        
top_rank_popularity_abs_df = pd.DataFrame(top_rank_popularity_abs_connections_dict)
top_rank_popularity_abs_df.index = top_rank_type_df.index.levels[0]

syn_popularity_abs_df = pd.DataFrame(syn_popularity_abs_connections_dict)
syn_popularity_abs_df.index = syn_type_df.index.levels[0]

popularity_abs_df = pd.DataFrame(popularity_abs_connections_dict)
popularity_abs_df.index = type_df.index.levels[0]

if sort_by == 'median_abs_count':
    syn_popularity_abs_df_zeros = syn_popularity_abs_df.fillna(0) # Replacing NaNs with zeros
    sorted_column_order = syn_popularity_abs_df.median().sort_values(ascending=False).index.tolist()

if sort_by == 'mean_abs_count':
    syn_popularity_abs_df_zeros = syn_popularity_abs_df.fillna(0) # Replacing NaNs with zeros
    sorted_column_order = syn_popularity_abs_df_zeros.mean().sort_values(ascending=False).index.tolist()

#################################### Neurotransmitter anaylsis ####################################
###################################################################################################
# This section creates matrices for all partners with NT information (type ad polarity)
# Using the following number code (relavant for heatmap plots or any quantitative analysis between excitation (E) and inhibition (I)):
# NT_code_dict = {'None' : 00 , 'GLUT' : 11, 'GABA' : 22, 'ACH' : 33, 'HIST' : 44, 'Exc.' : 1, 'Inh.' : -1}

#NT code dicttionary
NT_code_dict = {'None' : 1 ,'GLUT' : 2, 'GABA' : 3, 'ACH' : 4, 'HIST' : 5, 'Exc.' : 1, 'Inh.' : -1}
# Apply the custom function to create the new NT_type based on Flywire information
NT_df['NT_type'] = NT_df.apply(create_column_c, args=('From Flywire 1', 'From Flywire 2'), axis=1)
# Create a new column C using the map method with the dictionary
NT_df['NT_type_code'] = NT_df['NT_type'].map(NT_code_dict)
NT_df['EI_code'] = NT_df['Excitatory_Inhibitory'].map(NT_code_dict)

## Data frames with NT or EI label and code
NT_type_df = NT_df[['Neurons', 'NT_type','NT_type_code']].copy()
NT_polarity_df = NT_df[['Neurons', 'Excitatory_Inhibitory','EI_code']].copy()

## Creating a matrix of NT_type and EI codes for all parterns 
#  Create a new DataFrame with the same structure as syn_popularity_abs_df
#  but use the NT_type_code values from NT_df
syn_popularity_NT_type_df = syn_popularity_abs_df.copy()
syn_popularity_NT_polarity_df = syn_popularity_abs_df.copy()

# Filter the NT_df to include only rows with matching column names in syn_popularity_abs_df
matching_neurons = syn_popularity_abs_df.columns.intersection(NT_df['Neurons'])
filtered_NT_df = NT_df[NT_df['Neurons'].isin(matching_neurons)]

# Iterate over the rows in NT_df and update corresponding columns in syn_popularity_NT_type_df
for _, row in filtered_NT_df.iterrows():
    neuron = row['Neurons']
    nt_type = row['NT_type_code']
    nt_polarity = row['EI_code']
    syn_popularity_NT_type_df[neuron] = nt_type
    syn_popularity_NT_polarity_df[neuron] = nt_polarity

# Check for NaN values in syn_popularity_abs_df and set corresponding values in syn_popularity_NT_type_df to NaN
nan_mask = syn_popularity_abs_df.isnull()
syn_popularity_NT_type_df[nan_mask] = np.nan
syn_popularity_NT_polarity_df[nan_mask] = np.nan


########################################## Dicarding OUTLIERS ######################################
###################################################################################################
# Here we discard ouliers based on the IQR using the Tukey's fences method
# The IQR method calculates the range between the 25th and 75th percentiles of each column. 
# By defining a multiplier (often 1.5), you remove outliers (or replace them by NaNs) outside a certain range.

#Creating datasets without outliers
syn_popularity_rel_no_outliers_df = replace_outliers_with_nan(syn_popularity_rel_df)
syn_popularity_abs_no_outliers_df = replace_outliers_with_nan(syn_popularity_abs_df)



########################################## Defining sorting for plotting #####################################
#############################################################################################################

#Selecting for all neuron present in at least some proportion of columns (presence_threshold_neuron_filter) and keepind a certain order (sorted_column_order)
presence_threshold_sorted_column_order = [neuron for neuron in sorted_column_order if neuron in presence_threshold_neuron_filter]

if analyzing_cluster:
    # Create a new list containing the common elements in the same order as "user_defined_sorted_column_order"
    common_list = [item for item in user_defined_sorted_column_order if item in presence_threshold_sorted_column_order]
    presence_threshold_sorted_column_order = common_list

print(f'\nPresynaptic partners sorted and included:\n {presence_threshold_sorted_column_order}')

##########################################  ABSOLUTE SYNAPTIC COUNTS ##########################################
####################################   ELECTRON AND EXPANSION MIRCORSOPY   ################################### 

# Putting together ExM and EM data for specific presynaptic partners present in both


ExM_EM_absolut_counts = {}
for partner, curr_ExM_df in ExM_absolut_counts.items(): # for all neurons in ExM data set
    if partner in top_rank_popularity_abs_df.columns.tolist(): # for only neurons also in EM data set

        #Generating a compatible dataframe for EM
        curr_instance_post = top_rank_popularity_abs_df[partner].index.tolist()
        curr_count = top_rank_popularity_abs_df[partner].tolist()
        curr_EM_df = pd.DataFrame(list(zip(curr_instance_post, curr_count)),
               columns =['instance_post', 'W_new'])
        curr_EM_df['fly_ID'] = 'EM'
        curr_EM_df['type_pre'] = partner
        curr_EM_df['type_post'] = neuron_of_interest
        curr_EM_df['column_id'] = [string.split('::')[1] for string in curr_instance_post]
        curr_EM_df.dropna(how='any', axis=0, inplace=True)


        #Concatenating ExM and EM dataframes
        curr_ExM_EM_df = pd.concat([curr_ExM_df,curr_EM_df])
        curr_ExM_EM_df.reset_index(drop=True, inplace=True)
        ExM_EM_absolut_counts[partner] = curr_ExM_EM_df


############################################ PEARSON CORRELATION #############################################
#########################################  HIERARCHICAL CLUSTERING  ##########################################
##############################################################################################################

# Calculating statitstical significance for all correlations
# Correlation across columns between pair of neurons
# # Element-wise pearson correlation. Range: -1 to +1

### Relative counts
curr_df = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
curr_df = curr_df.fillna(0).copy()
correlation_rel_no_NaN_df, p_values_correlation_rel_no_NaN_df = calculate_correlation_and_p_values(curr_df)
p_values_correlation_rel_no_NaN_df_asterix_df = p_values_correlation_rel_no_NaN_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))
### Absolute counts
curr_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
curr_df = curr_df.fillna(0).copy()
correlation_abs_no_NaN_df, p_values_correlation_abs_no_NaN_df = calculate_correlation_and_p_values(curr_df)
p_values_correlation_abs_no_NaN_df_asterix_df = p_values_correlation_abs_no_NaN_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

#Hierarchical clustering for correlations
# Perform hierarchical clustering
dendrogram_pearson = hierarchy.linkage(correlation_abs_no_NaN_df.to_numpy(), method='ward')
pearson_order = hierarchy.leaves_list(dendrogram_pearson)
# Create a new DataFrame with reordered rows and columns
correlation_abs_no_NaN_reordered = correlation_abs_no_NaN_df.iloc[pearson_order].copy()
correlation_abs_no_NaN_reordered = correlation_abs_no_NaN_reordered.iloc[:,pearson_order].copy()
# Calculate cosine similarity
pearson_abs_reordered = cosine_similarity(correlation_abs_no_NaN_reordered.values)
    
    
# Some sorting based on correlation values
# For relative counts
column_order = correlation_rel_no_NaN_df.sum().sort_values(ascending=False).index.tolist() # new column order based on sum (it will create a gradien from most-correlated to most.anticorrelated)
sorted_correlation_rel_no_NaN_df= correlation_rel_no_NaN_df[column_order] # swpapping columns
sorted_p_values_correlation_rel_no_NaN_df_asterix_df = p_values_correlation_rel_no_NaN_df_asterix_df[column_order]  # swpapping columns
# For absolute counts
column_order = correlation_abs_no_NaN_df.sum().sort_values(ascending=False).index.tolist() # new column order based on sum (it will create a gradien from most-correlated to most.anticorrelated)
sorted_correlation_abs_no_NaN_df= correlation_abs_no_NaN_df[column_order] # swpapping columns
sorted_p_values_correlation_abs_no_NaN_df_asterix_df = p_values_correlation_abs_no_NaN_df_asterix_df[column_order]  # swpapping columns


# Removing the +1 correlated diagonal (setting it to NaN)
# For relative counts
correlation_rel_no_NaN_df.replace(1.0, np.NaN, inplace = True)
sorted_correlation_rel_no_NaN_df.replace(1.0, np.NaN, inplace = True)
# For absolute coutns
correlation_abs_no_NaN_df.replace(1.0, np.NaN, inplace = True)
sorted_correlation_abs_no_NaN_df.replace(1.0, np.NaN, inplace = True)


############################################   PERMUTATION TEST   ###########################################
############################################  PEARSON CORRELATIONS ##########################################
#TODO SO far only done for correlation related to abs counts. Do it eventually also for relative counts

if permutation_analysis:
    if relative_or_absolute == 'absolute-counts':
        ### Absolute counts
        # Data
        curr_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
        curr_df = curr_df.fillna(0).copy()
        if analyzing_cluster:
            curr_dataset_abs_df = dataset_abs_df.fillna(0).copy()
        else:
            curr_dataset_abs_df = curr_df.copy() 

        # Columns to be compared
        column_names = curr_df.columns.tolist()

        # In case you want to run it for different subsets of the data, code need to be modify for defining column_names
        clusters_list = [curr_df]  # Insert your actual cluster DataFrames

        # Initialize empty DataFrames to store observed correlations and p-values
        observed_corr_abs_df = pd.DataFrame(columns=column_names, index=column_names)
        p_value_permutation_abs_df  = pd.DataFrame(columns=column_names, index=column_names)

        # Perform permutation test for each cluster and each pair of columns
        for i, cluster_df in enumerate(clusters_list):
            #print(f"Cluster {i + 1}:")
            print(f"\nDoing permutation test for all pairs...\n")
            for column_pair in combinations(column_names, 2):
                column1_name, column2_name = column_pair
                #print(f"Pair: {column1_name} and {column2_name}")
                observed_corr, p_value, shuffled_corrs = permutation_test(cluster_df, curr_dataset_abs_df, column1_name, column2_name, num_permutations, _seed)
                
                # Save observed correlation and p-value in the corresponding DataFrames
                observed_corr_abs_df.loc[column1_name, column2_name] = observed_corr
                observed_corr_abs_df.loc[column2_name, column1_name] = observed_corr
                p_value_permutation_abs_df .loc[column1_name, column2_name] = p_value
                p_value_permutation_abs_df .loc[column2_name, column1_name] = p_value

        observed_corr_abs_df = observed_corr_abs_df.apply(pd.to_numeric, errors='coerce')
        p_value_permutation_abs_df  = p_value_permutation_abs_df .apply(pd.to_numeric, errors='coerce')


        ### Keeping the p-values from the pearson correlations only if the permutation test was passed:
        # Create a boolean mask where values in p_value_permutation_abs_df  are less than 0.05
        mask_permutation = p_value_permutation_abs_df  < 0.05
        mask_correlation = p_values_correlation_abs_no_NaN_df < 0.05
        # Create a new DataFrame with the same columns and indexes as p_values_correlation_abs_no_NaN_df
        p_values_correlation_abs_after_permutation_df = pd.DataFrame(index=p_values_correlation_abs_no_NaN_df.index,
                                                                columns=p_values_correlation_abs_no_NaN_df.columns)
        # Set the values in the new DataFrame to "" where the mask is False (values in p_value_permutation_abs_df  >= 0.05)
        p_values_correlation_abs_after_permutation_df[~mask_permutation] = ""
        # Set the values in the new DataFrame to the values in p_values_correlation_abs_no_NaN_df where the mask is True
        p_values_correlation_abs_after_permutation_df[mask_permutation] = p_values_correlation_abs_no_NaN_df[mask_permutation]
        # Set the values in the new dataFrame to '"" where the mask is False
        p_values_correlation_abs_after_permutation_df[~mask_correlation] = ""


    elif relative_or_absolute == 'relative-counts':
        ### Relative counts
        # Data
        curr_df = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
        curr_df = curr_df.fillna(0).copy()
        if analyzing_cluster:
            curr_dataset_rel_df = dataset_rel_df.fillna(0).copy()
        else:
            curr_dataset_rel_df = curr_df.copy() 

        # Columns to be compared
        column_names = curr_df.columns.tolist()

        # In case you want to run it for different subsets of the data, code need to be modify for defining column_names
        clusters_list = [curr_df]  # Insert your actual cluster DataFrames

        # Initialize empty DataFrames to store observed correlations and p-values
        observed_corr_rel_df = pd.DataFrame(columns=column_names, index=column_names)
        p_value_permutation_rel_df  = pd.DataFrame(columns=column_names, index=column_names)

        # Perform permutation test for each cluster and each pair of columns
        for i, cluster_df in enumerate(clusters_list):
            #print(f"Cluster {i + 1}:")
            print(f"\nDoing permutation test for all pairs...\n")
            for column_pair in combinations(column_names, 2):
                column1_name, column2_name = column_pair
                #print(f"Pair: {column1_name} and {column2_name}")
                observed_corr, p_value, shuffled_corrs = permutation_test(cluster_df, curr_dataset_rel_df, column1_name, column2_name, num_permutations, _seed)
                
                # Save observed correlation and p-value in the corresponding DataFrames
                observed_corr_rel_df.loc[column1_name, column2_name] = observed_corr
                observed_corr_rel_df.loc[column2_name, column1_name] = observed_corr
                p_value_permutation_rel_df .loc[column1_name, column2_name] = p_value
                p_value_permutation_rel_df .loc[column2_name, column1_name] = p_value

        observed_corr_rel_df = observed_corr_rel_df.apply(pd.to_numeric, errors='coerce')
        p_value_permutation_rel_df  = p_value_permutation_rel_df .apply(pd.to_numeric, errors='coerce')


        ### Keeping the p-values from the pearson correlations only if the permutation test was passed:
        # Create a boolean mask where values in p_value_permutation_abs_df  are less than 0.05
        mask_permutation = p_value_permutation_rel_df  < 0.05
        mask_correlation = p_values_correlation_rel_no_NaN_df < 0.05
        # Create a new DataFrame with the same columns and indexes as p_values_correlation_abs_no_NaN_df
        p_values_correlation_rel_after_permutation_df = pd.DataFrame(index=p_values_correlation_rel_no_NaN_df.index,
                                                                columns=p_values_correlation_rel_no_NaN_df.columns)
        # Set the values in the new DataFrame to "" where the mask is False (values in p_value_permutation_abs_df  >= 0.05)
        p_values_correlation_rel_after_permutation_df[~mask_permutation] = ""
        # Set the values in the new DataFrame to the values in p_values_correlation_abs_no_NaN_df where the mask is True
        p_values_correlation_rel_after_permutation_df[mask_permutation] = p_values_correlation_rel_no_NaN_df[mask_permutation]
        # Set the values in the new dataFrame to '"" where the mask is False
        p_values_correlation_rel_after_permutation_df[~mask_correlation] = ""



############################################   COSINE SIMILARITY    ###########################################
############################################ HIERARCHICAL CLUSTERING ##########################################
###############################################################################################################

# Data
if relative_or_absolute == 'absolute-counts':
    _data = top_rank_popularity_abs_df[presence_threshold_sorted_column_order].copy()
elif relative_or_absolute == 'relative-counts':
    _data = top_rank_popularity_rel_df[presence_threshold_sorted_column_order].copy()

if not d_v_filter:
    cosine_sim_df, cosine_sim_summary_df, cosine_row_order, dendrogram_cosine, cosine_sim_reordered_df, _data_reordered_cosine_sim, cosine_sim, cosine_sim_reordered, cos_sim_medians = cosine_similarity_and_clustering(_data,cosine_subgroups)


###################################### DENDROGRAM CLUSTERING ######################################
###################################################################################################

#TODO: define clusters based on branches of the dendogram using one of the options:
# - Elbow Method
# - Gap Statistic
# - Silhouette Analysis (DONE)

if all([cluster_with_dendrogram, not d_v_filter]):

    # Range of clusters to consider
    range_n_clusters = range(8, 10) # prevously used: range(4, 10)
    # List to store silhouette scores
    silhouette_scores = []
    # Calculate silhouette scores for different numbers of clusters
    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(cosine_sim)
        silhouette_avg = silhouette_score(cosine_sim, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters with the highest silhouette score
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    # Perform hierarchical clustering with the optimal number of clusters
    clusterer = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(cosine_sim)
    # Add the "cluster" column to the "cosine_sim_summary_df" dataframe
    cosine_sim_summary_df['cluster'] = cluster_labels

    # Extract lists of indexes names (postsynaptic colums IDs) for the differe clusters

    clusters_optic_lobe_ids = {}  # Dictionary to store clusters and their corresponding index names

    for index, row in cosine_sim_summary_df.iterrows():
        index_parts = index.split(':')
        row_cluster_id = row['cluster']
        if row_cluster_id not in clusters_optic_lobe_ids:
            clusters_optic_lobe_ids[row_cluster_id] = []
        clusters_optic_lobe_ids[row_cluster_id].append(index_parts[2])

    # Convert the dictionary values to sets to remove duplicates, then back to lists
    for row_cluster_id in clusters_optic_lobe_ids:
        clusters_optic_lobe_ids[row_cluster_id] = list(set(clusters_optic_lobe_ids[row_cluster_id]))

    # Print the result
    for row_cluster_id, index_names in clusters_optic_lobe_ids.items():
        print(f'Cluster {row_cluster_id}: {index_names}')

    # Save cluster lists as text files
    if save_clusters_txt: 
        for row_cluster_id, index_names in clusters_optic_lobe_ids.items():
            file_name = f"\Tm9_cosine_similarity_cluster_{row_cluster_id}_{optic_lobe}.txt"
            df_cluster = pd.DataFrame(index_names, columns=['Index Name'])
            df_cluster.to_csv(txtPath+file_name, sep='\t', index=False, header=False)


    # Print the optimal number of clusters
    print('\nSilhouette analysis:')
    print(f"\nOptimal number of clusters from cosine similarity based on dendrogram: {optimal_n_clusters}")

    ## Separating cosine_sim values in clusters
    unique_clusters = np.unique(cluster_labels)
    # Create a dictionary to store the arrays for each cluster
    cosine_cluster_arrays = {}
    # Iterate over each unique cluster label
    for cluster_label in unique_clusters:
        # Find the indices of rows and columns belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        # Get the values of the current cluster from the cosine_sim array
        cluster_array = cosine_sim[cluster_indices][:, cluster_indices]
        # Store the cluster array in the dictionary with the cluster label as the key
        cosine_cluster_arrays[cluster_label] = cluster_array
    # Now "cosine_cluster_arrays" contains all the 2D arrays corresponding to each cluster
    # You can access them using the cluster label as the key, e.g., cosine_cluster_arrays[0] for the first cluster



############################################# COEFICIENT OF VARIATION #########################################
###############################################################################################################

# Table for Coefficient of variation calculations

#Consider filtering some columns (here presynaptic neurons) or indexes that are not interested
curr_rel_stats_df  = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
curr_abs_stats_df  = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
curr_rel_stats_no_ouliers_df  = syn_popularity_rel_no_outliers_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
curr_abs_stats_no_ouliers_df  = syn_popularity_abs_no_outliers_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"

#Calculate basic statistcs
curr_rel_stats_df = curr_rel_stats_df[curr_rel_stats_df.max().sort_values(ascending = False).index].describe()
curr_abs_stats_df = curr_abs_stats_df[curr_abs_stats_df.max().sort_values(ascending = False).index].describe()
curr_rel_stats_no_ouliers_df = curr_rel_stats_no_ouliers_df[curr_rel_stats_no_ouliers_df.max().sort_values(ascending = False).index].describe()
curr_abs_stats_no_ouliers_df = curr_abs_stats_no_ouliers_df[curr_abs_stats_no_ouliers_df.max().sort_values(ascending = False).index].describe()

# Calculate coefficient of variation (Z-score)
curr_rel_stats_df.loc['C.V.'] = curr_rel_stats_df.loc['std'] / curr_rel_stats_df.loc['mean']
curr_abs_stats_df.loc['C.V.'] = curr_abs_stats_df.loc['std'] / curr_abs_stats_df.loc['mean']
curr_rel_stats_no_ouliers_df.loc['C.V.'] = curr_rel_stats_no_ouliers_df.loc['std'] / curr_rel_stats_no_ouliers_df.loc['mean']
curr_abs_stats_no_ouliers_df.loc['C.V.'] = curr_abs_stats_no_ouliers_df.loc['std'] / curr_abs_stats_no_ouliers_df.loc['mean']

# Calculate mean of all statistics
curr_rel_stats_df['mean'] = curr_rel_stats_df.mean(axis=1)
curr_abs_stats_df['mean'] = curr_abs_stats_df.mean(axis=1)
curr_rel_stats_no_ouliers_df['mean'] = curr_rel_stats_no_ouliers_df.mean(axis=1)
curr_abs_stats_no_ouliers_df['mean'] = curr_abs_stats_no_ouliers_df.mean(axis=1)



######################################### DIMENTIONALITY Reduction #######################################
################################################   PCA   #################################################
# #For PCA plot in figure 5
# order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20','C3','Tm1','PS125','L4','ML1','TmY17','C2','OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
# rel_data = syn_popularity_rel_df[order].copy()

#For relative counts:
# Data
rel_data = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy()
rel_data= rel_data.fillna(0)
rel_data_array = rel_data.to_numpy(dtype=int,copy=True)

### PCA

## Standardize
rel_data_array_norm = rel_data_array-rel_data_array.mean(axis=0) #Standardize the features (row axis). We want mean across columns
# Calculate standard deviation, handling potential division by zero or NaN
std_dev = rel_data_array_norm.std(axis=0)
#std_dev[np.isnan(std_dev) | (std_dev == 0)] = 1.0
# Divide by standard deviation, handling potential division by zero or NaN
rel_data_array_norm /= std_dev
n = rel_data_array_norm.shape[0]

## Cov matrix and eigenvectors
rel_cov = (1/n) * rel_data_array_norm.T @ rel_data_array_norm #It shoould have the dimentions of your features: (dim(features),dim(features))
rel_eigvals, rel_eigvecs = np.linalg.eig(rel_cov)
k = np.argsort(rel_eigvals)[::-1]
rel_eigvals = rel_eigvals[k]
rel_eigvecs = rel_eigvecs[:,k]

# For absolute counts:
# Data
abs_data = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy()
abs_data= abs_data.fillna(0)
abs_data_array = abs_data.to_numpy(dtype=int,copy=True)

#### PCA
## Standardize
abs_data_array_norm = abs_data_array-rel_data_array.mean(axis=0) #Standardize the features (row axis). We want mean across columns
# Calculate standard deviation, handling potential division by zero or NaN
std_dev = abs_data_array_norm.std(axis=0)
#std_dev[np.isnan(std_dev) | (std_dev == 0)] = 1.0
# Divide by standard deviation, handling potential division by zero or NaN
abs_data_array_norm /= std_dev
n = abs_data_array_norm.shape[0]

## Cov matrix and eigenvectors
abs_cov = (1/n) * abs_data_array_norm.T @ abs_data_array_norm #It shoould have the dimentions of your features: (dim(features),dim(features))
abs_eigvals, abs_eigvecs = np.linalg.eig(abs_cov)

#Taking the real part of the eigenvectors if it is a complex number
abs_eigvecs = np.real(abs_eigvecs)

k = np.argsort(abs_eigvals)[::-1]
abs_eigvals = abs_eigvals[k]
abs_eigvecs = abs_eigvecs[:,k]


######################################## SPATIAL DISTRIBUTION ANALYSIS #######################################
####################################### Nearest neighbour distribution #######################################

# Nearest neighbour distance distribution of presynaptic partner connection to Tm9s
_data = binary_df[presence_threshold_sorted_column_order] # FIlter abnd sorting
pre_partner_list = _data.columns.tolist()

#Extracting the XYZ locations of postsynpatic neurons of a given presynaptic partner
for pre_partner in pre_partner_list:
    curr_df = df[(df['type_pre'] == pre_partner) & (df['W_new'] >= 3) ].copy()
    curr_post_neurons_IDs = curr_df['optic_lobe_id'].unique().tolist()
    ##Gettting the center point in specific neuropile from database
    xyz_neuropil = 'XYZ-ME'
    xyz_df = database_df[database_df['optic_lobe_id'].isin(curr_post_neurons_IDs)].copy()
    xyz_pre = xyz_df[xyz_neuropil].tolist()
    # Split each string by comma and convert the elements to floats
    xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
    xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])
    xyz_pre_arr_new = xyz_pre_arr_new /1000 # Changing nanometers to micrometers

    ##Calculating the Nearest neighbour ditances per each point
    # Calculate pairwise distances
    data =  xyz_pre_arr_new
    distances = cdist(data, data)
    # Set diagonal elements to a high value (as each point is its own nearest neighbor)
    np.fill_diagonal(distances, np.inf)
    # Find the minimum distance for each point
    nearest_distances = np.min(distances, axis=1)
    #print(f'{pre_partner}: {len(nearest_distances)} data points')

    # CONTINUE THE ANALYSIS AND PERFORM STATISTICAL COMPARISON
    #TODO plot and save
    
    # # Plotting the distributions
    # plt.hist(nearest_distances, bins=40, edgecolor='black')
    # plt.xlabel('Nearest Neighbor Distance (um)')
    # plt.ylabel('Frequency')
    # plt.title(f'Distribution of Nearest Neighbor Distances for {pre_partner}')
    # plt.show()


#%%
######################################## PRINT STATEMENTs #########################################
## Generating text
counts_mean = syn_df.groupby(['instance_post']).agg({'W_new': sum}).mean()[0]
counts_std = syn_df.groupby(['instance_post']).agg({'W_new': sum}).std()[0]
final_synaptic_counts = f'{counts_mean:.2f} ± {counts_std:.2f}'
partners_number_mean = syn_df.groupby(['instance_post']).agg({'type_pre':'count'}).mean()[0]
partners_number_std = syn_df.groupby(['instance_post']).agg({'type_pre':'count'}).std()[0]
final_partners_number = f'{partners_number_mean:.2f} ± {partners_number_std:.2f}'
cell_types_number_mean = syn_df.groupby(['instance_post']).agg({'type_pre':'nunique'}).mean()[0]
cell_types_number_std = syn_df.groupby(['instance_post']).agg({'type_pre':'nunique'}).std()[0]
final_cell_types_number = f'{cell_types_number_mean:.2f} ± {cell_types_number_std:.2f}'

print(f'Final numbers in our final coverage {final_coverage} (mean+-std): \n')
print(f'Final #of contacts: {final_synaptic_counts}(mean+-std)(n={len(id_column)})') # final number of contacts (mean+-std) after filtering
print(f'Final #of presynaptic cells: {final_partners_number}(mean+-std)(n={len(id_column)})') # final number of presynaptic cells (mean+-std) after filtering
print(f'Final #of presynaptic cell types {final_cell_types_number}(mean+-std)(n={len(id_column)})') # final number of presynaptic cell types (mean+-std) after filtering
print('\n')

#TODO Print absolute contacts (mean + std) for the excluded_partners combined
if discard_homogeneous:
    ## For relative counts
    # Calculate the sum, mean, std, and SEM for each row
    excluded_partners_df = top_rank_popularity_rel_df[excluded_partners].copy()
    excluded_partners_df['sum'] = excluded_partners_df.sum(axis=1)
    excluded_partners_df['mean'] = excluded_partners_df.mean(axis=1)
    excluded_partners_df['std'] = excluded_partners_df.std(axis=1)
    excluded_partners_df['median'] = excluded_partners_df.median(axis=1)
    # Calculate the SEM
    num_data_points =excluded_partners_df.shape[1]  # Number of columns in the DataFrame
    excluded_partners_df['sem'] = excluded_partners_df['std'] / np.sqrt(num_data_points)
    # Calculate the mean and standard deviation for each column
    column_means = excluded_partners_df.mean(axis=0)
    column_medians = excluded_partners_df.median(axis=0)
    column_stds = excluded_partners_df.std(axis=0)
    column_sems = column_stds / np.sqrt(excluded_partners_df.shape[0])
    print(f'For this excluded partners: {excluded_partners} ') 
    print(f"Input coverage in relative (%) synaptic number being excluded: {round(column_means['sum'],2)}+-{round(column_stds['sum'], 2)} or {round(column_sems['sum'], 2)} (mean+-std or sem)")

    ## For absolute counts
    # Calculate the sum, mean, std, and SEM for each row
    excluded_partners_df = top_rank_popularity_abs_df[excluded_partners].copy()
    excluded_partners_df['sum'] = excluded_partners_df.sum(axis=1)
    excluded_partners_df['mean'] = excluded_partners_df.mean(axis=1)
    excluded_partners_df['std'] = excluded_partners_df.std(axis=1)
    excluded_partners_df['median'] = excluded_partners_df.median(axis=1)
    # Calculate the SEM
    num_data_points =excluded_partners_df.shape[1]  # Number of columns in the DataFrame
    excluded_partners_df['sem'] = excluded_partners_df['std'] / np.sqrt(num_data_points)

    # Calculate the mean and standard deviation for each column
    column_means = excluded_partners_df.mean(axis=0)
    column_medians = excluded_partners_df.median(axis=0)
    column_stds = excluded_partners_df.std(axis=0)
    column_sems = column_stds / np.sqrt(excluded_partners_df.shape[0])
    # # Add the means and stds as new rows to the DataFrame
    # excluded_partners_df = excluded_partners_df.append(column_means, ignore_index=True)
    # excluded_partners_df = excluded_partners_df.append(column_stds, ignore_index=True)
    # # Assign row labels for clarity
    # excluded_partners_df.index = excluded_partners_df.index.to_list()[:-2] + ['mean', 'std']
    print(f"Input coverage in absolute synaptic number being excluded: {round(column_means['sum'],2)}+-{round(column_stds['sum'], 2)} or {round(column_sems['sum'], 2)} (mean+-std or sem)")  


#%%
############################################ SAVING SECTION #############################################
#########################################################################################################
#########################################################################################################


## Saving processed data
if saving_processed_data:
    ## Saving data frames
    #Absolute count in an excel file
    file_name = f'{fileName_txt}_{dataset_name}_{fileDate}.xlsx'
    savePath = os.path.join(output_dataPath, file_name)
    top_rank_popularity_abs_df.to_excel(savePath, sheet_name='Absolut_counts')

    #More dataframes in same excel file
    book = load_workbook(savePath)
    writer = pd.ExcelWriter(savePath, engine = 'openpyxl')
    writer.book = book
    top_rank_popularity_rel_df.to_excel(writer, sheet_name='Relative_counts')
    binary_df.to_excel(writer, sheet_name='Binary')
    rank_df.to_excel(writer, sheet_name='Rank')
    writer.save()
    writer.close()
    print('Processed data saved')

    ## Saving summary text

    #TODO Save the following texts
    final_coverage # final % coverage (mean+-std) after filtering
    final_synaptic_counts # final number of contacts (mean+-std) after filtering
    final_partners_number # final number of presynaptic cells (mean+-std) after filtering
    final_cell_types_number # final number of presynaptic cell types (mean+-std) after filtering

    


#%% 
############################################# PLOTTING SECTION ##############################################
#############################################################################################################
#############################################################################################################

if plot_data:



    ################################################# BAR - PLOTS ##############################################
    ############################################################################################################

    # Itinial barplots for non-filtered and filtered data about total number of synapses and neurons per column

    ######################################## Total number of synapses ##########################################


    #Data
    syn_df_grouped = syn_df.groupby(['instance_post']).agg({'W_new': sum})
    df_0_grouped = df_0.groupby(['instance_post']).agg({'W_new': sum})
    data_label = 'Synaptic contacts'
    data_variable = 'W_new'

    # Create a 2x2 grid of subplots
    fig= plt.figure(figsize=(10, 8))
    G = gridspec.GridSpec(2, 2)

    # Boxplot for df_0
    boxprops_df_0 = dict(color='blue')
    box_plot = plt.subplot(G[0, 0])
    box_plot.boxplot(df_0_grouped[data_variable], positions=[0], widths=0.4, showmeans=False, boxprops=boxprops_df_0)
    boxprops_syn_df = dict(color='orange')
    box_plot.boxplot(syn_df_grouped[data_variable], positions=[0.6], widths=0.4, showmeans=False, boxprops=boxprops_syn_df)

    # Set the font size of y and x labels and ticks for boxplots
    box_plot.set_ylabel(data_label, fontsize=12)
    box_plot.set_xlabel(dataset_name, fontsize=12)
    box_plot.tick_params(axis='both', which='both', labelsize=10)
    box_plot.grid(False)
    box_plot.spines['right'].set_visible(False)
    box_plot.spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    box_plot.set_xticks([0, 0.6])
    box_plot.set_xticklabels(['all syn', 'syn >=3'])


    # Plot histogram in the second row, second column
    hist_plot = plt.subplot(G[0,1])
    hist_plot.hist([df_0_grouped[data_variable], syn_df_grouped[data_variable]], bins=10, label=['all syn', 'syn >=3'])
    hist_plot.set_xlabel(data_label, fontsize=12)
    hist_plot.tick_params(axis='both', which='both', labelsize=10)
    hist_plot.set_ylabel('Frequency', fontsize=12)
    hist_plot.legend()


    # Plot bar plot in the second row, first column
    bar_plot = plt.subplot(G[1,:])
    bar_plot.bar(df_0_grouped.index, df_0_grouped[data_variable], label='all syn')
    bar_plot.bar(syn_df_grouped.index, syn_df_grouped[data_variable], label='syn >=3')
    bar_plot.set_ylabel(data_label, fontsize=12)
    bar_plot.tick_params(axis='both', which='both', labelsize=10)
    bar_plot.set_xlabel(dataset_name, fontsize=12)
    bar_plot.set_xlabel('Columns', fontsize=12)
    bar_plot.set_xticklabels([])
    bar_plot.legend()



    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Total-number-synapses_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Itinial barplots: Synaptic contacts')
    plt.close(fig)


    ######################################## Total number of presynaptic partners ##########################################

    #Data
    syn_df_grouped = syn_df.groupby(['instance_post']).agg({'type_pre':'count'})
    df_0_grouped = df_0.groupby(['instance_post']).agg({'type_pre':'count'})
    data_label = 'Presynaptic partners / segments'
    data_variable = 'type_pre'

    # Create a 2x2 grid of subplots
    fig= plt.figure(figsize=(10, 8))
    G = gridspec.GridSpec(2, 2)

    # Boxplot for df_0
    boxprops_df_0 = dict(color='blue')
    box_plot = plt.subplot(G[0, 0])
    box_plot.boxplot(df_0_grouped[data_variable], positions=[0], widths=0.4, showmeans=False, boxprops=boxprops_df_0)
    boxprops_syn_df = dict(color='orange')
    box_plot.boxplot(syn_df_grouped[data_variable], positions=[0.6], widths=0.4, showmeans=False, boxprops=boxprops_syn_df)

    # Set the font size of y and x labels and ticks for boxplots
    box_plot.set_ylabel(data_label, fontsize=12)
    box_plot.set_xlabel(dataset_name, fontsize=12)
    box_plot.tick_params(axis='both', which='both', labelsize=10)
    box_plot.grid(False)
    box_plot.spines['right'].set_visible(False)
    box_plot.spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    box_plot.set_xticks([0, 0.6])
    box_plot.set_xticklabels(['all syn', 'syn >=3'])


    # Plot histogram in the second row, second column
    hist_plot = plt.subplot(G[0,1])
    hist_plot.hist([df_0_grouped[data_variable], syn_df_grouped[data_variable]], bins=10, label=['all syn', 'syn >=3'])
    hist_plot.set_xlabel(data_label, fontsize=12)
    hist_plot.tick_params(axis='both', which='both', labelsize=10)
    hist_plot.set_ylabel('Frequency', fontsize=12)
    hist_plot.legend()


    # Plot bar plot in the second row, first column
    bar_plot = plt.subplot(G[1,:])
    bar_plot.bar(df_0_grouped.index, df_0_grouped[data_variable], label='all syn')
    bar_plot.bar(syn_df_grouped.index, syn_df_grouped[data_variable], label='syn >=3')
    bar_plot.set_ylabel(data_label, fontsize=12)
    bar_plot.tick_params(axis='both', which='both', labelsize=10)
    bar_plot.set_xlabel(dataset_name, fontsize=12)
    bar_plot.set_xlabel('Columns', fontsize=12)
    bar_plot.set_xticklabels([])
    bar_plot.legend()

    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Total-number-presynaptic-neurons_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Itinial barplots: Presynaptic neurons')
    plt.close(fig)

    ######################################## Total number of presynaptic cell types ##########################################

    #Data
    syn_df_grouped = syn_df.groupby(['instance_post']).agg({'type_pre':'nunique'})
    df_0_grouped = df_0.groupby(['instance_post']).agg({'type_pre':'nunique'})
    data_label = 'Presynaptic cell types / segments'
    data_variable = 'type_pre'

    # Create a 2x2 grid of subplots
    fig= plt.figure(figsize=(10, 8))
    G = gridspec.GridSpec(2, 2)

    # Boxplot for df_0
    boxprops_df_0 = dict(color='blue')
    box_plot = plt.subplot(G[0, 0])
    box_plot.boxplot(df_0_grouped[data_variable], positions=[0], widths=0.4, showmeans=False, boxprops=boxprops_df_0)
    boxprops_syn_df = dict(color='orange')
    box_plot.boxplot(syn_df_grouped[data_variable], positions=[0.6], widths=0.4, showmeans=False, boxprops=boxprops_syn_df)

    # Set the font size of y and x labels and ticks for boxplots
    box_plot.set_ylabel(data_label, fontsize=12)
    box_plot.set_xlabel(dataset_name, fontsize=12)
    box_plot.tick_params(axis='both', which='both', labelsize=10)
    box_plot.grid(False)
    box_plot.spines['right'].set_visible(False)
    box_plot.spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    box_plot.set_xticks([0, 0.6])
    box_plot.set_xticklabels(['all syn', 'syn >=3'])


    # Plot histogram in the second row, second column
    hist_plot = plt.subplot(G[0,1])
    hist_plot.hist([df_0_grouped[data_variable], syn_df_grouped[data_variable]], bins=10, label=['all syn', 'syn >=3'])
    hist_plot.set_xlabel(data_label, fontsize=12)
    hist_plot.tick_params(axis='both', which='both', labelsize=10)
    hist_plot.set_ylabel('Frequency', fontsize=12)
    hist_plot.legend()


    # Plot bar plot in the second row, first column
    bar_plot = plt.subplot(G[1,:])
    bar_plot.bar(df_0_grouped.index, df_0_grouped[data_variable], label='all syn')
    bar_plot.bar(syn_df_grouped.index, syn_df_grouped[data_variable], label='syn >=3')
    bar_plot.set_ylabel(data_label, fontsize=12)
    bar_plot.tick_params(axis='both', which='both', labelsize=10)
    bar_plot.set_xlabel(dataset_name, fontsize=12)
    bar_plot.set_xlabel('Columns', fontsize=12)
    bar_plot.set_xticklabels([])
    bar_plot.legend()


    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Total-number-presynaptic-types_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Itinial barplots: Presynaptic types')
    plt.close(fig)





    ##################################    Bar plots showing presence and absence of neuron partners  ###############################
    #Figure
    fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (40*cm, 15*cm))
    fig.tight_layout(pad=10) # Adding some space between subplots

    color_absent = [204/255,236/255,230/255]
    color_present = [27/255,168/255,119/255]
    color_present = sns.color_palette(hex_color, as_cmap=False)[-1] # 



    # First axis
    sorted_abs_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                        edgecolor = None, ax = axs[0],legend=False)
    axs[0].set_title(f'{neuron_of_interest}, Presence / absence across columns, syn>={min_desired_count}')
    #axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[0].set_xlabel('Presynaptic neuron')
    axs[0].set_ylabel('Number of columns')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)


    # Next axis
    sorted_rel_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                        edgecolor = None, ax = axs[1],legend=False)
    axs[1].set_title(f'{neuron_of_interest}, Presence / absence across columns, syn>={min_desired_count}')
    #axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[1].set_xlabel('Presynaptic neuron')
    axs[1].set_ylabel('% of columns')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

    # Next axis
    sorted_thr_rel_presence_absence_df['Presynaptic neuron_cat'] = pd.Categorical(
        sorted_thr_rel_presence_absence_df['Presynaptic neuron'], 
        categories=presence_threshold_sorted_column_order, 
        ordered=True
    )
    sorted_thr_rel_presence_absence_df.sort_values('Presynaptic neuron_cat', inplace = True)
    sorted_thr_rel_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                        edgecolor = "black", ax = axs[2],legend=False)
    axs[2].set_title(f'{neuron_of_interest}, Presence / absence across columns above {presence_threshold}, syn>={min_desired_count}, sorted by {sort_by}')
    #axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[2].set_xlabel('Presynaptic neuron')
    axs[2].set_ylabel('% of columns')
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Percentage_columns_partner_presence_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Percentatge across columns plotted and saved')
    plt.close(fig)


    ###########################################STACKED BAR PLOTS################################################

    #TODO Check if this code works
    if stacked_plots:
        #syn_df.to_csv(r'C:\Users\jcornean\PhD\FAFB\FlyWire\Excels\syn_df_Tm1S.csv')
        _data = nan_popularity_rel_df.fillna(0)

        mean_input = []
        for neuron in (list(_data.columns)):
            curr_input = _data[neuron].mean()
            mean_input.append(curr_input)
        tm_rel_dict = dict(map(lambda i,j : (i,j),(list(_data.columns)), mean_input))
        tm_rel_df = pd.DataFrame(tm_rel_dict, index = [neuron_of_interest])
        tm_rel_df = tm_rel_df.sort_values(by=neuron_of_interest, axis=1, ascending = False)
        column_to_move = tm_rel_df.pop("N.I.") #replace N.I column to the end of the df
        tm_rel_df['N.I.'] = column_to_move

        #tm_rel_df.to_csv(r'C:\Users\jcornean\PhD\FAFB\FlyWire\Excels\Tm1_relative_counts_mean.csv')

        #only one Tm for the percentage
        #plt.figure(figsize=(6, 12))
        random.seed(97)
        c_names = list(mcolors.CSS4_COLORS.values())
        random.shuffle(c_names)
        ax = tm_rel_df.plot(kind='bar', stacked=True, color=c_names)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend (handles[:5], labels[:5] , ncol = 1, loc = "center left", frameon = False, bbox_to_anchor=(0.8, 0.5))
        plt.xlabel(neuron_of_interest)
        plt.ylabel('% synapses')
        plt.title(f'Percent of Synapses of Input neurons to {neuron_of_interest}')
        #Plot saving
        if save_figures:
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Stacked_relative_counts_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
            plt.savefig(save_path+figure_title)
        plt.close()


    #%% 
    ############################################ BOX - PLOTS -BY CATEGORY ######################################
    ############################################################################################################


    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(20*cm, 20*cm))
    fig.suptitle(f'Comparison between {category_column}')
    categories = syn_df[category_column].unique()
    positions = np.arange(len(categories))
    # Define a color palette
    color_palette = sns.color_palette(color_cat_set, len(categories))

    ################### Data first axis: Synaptic counts
    # Iterate over unique categories 
    for i, category in enumerate(categories):
        # Filter the data for the current category
        category_data = syn_df[syn_df[category_column] == category]
        
        # Compute grouped data for the current category
        category_grouped = category_data.groupby(['instance_post']).agg({'W_new': sum})
        
        # Plot the boxplot for the current category with the specified color
        box = axs[0, 0].boxplot(category_grouped['W_new'], positions=[positions[i]], widths=0.4, showmeans=True, patch_artist=True)

        # Customize the box color
        for patch in box['boxes']:
            patch.set_facecolor(color_palette[i])
        
        # Customize the outlier markers
        for patch in box['fliers']:
            patch.set(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3)  # Black-filled rhombus

        # Add mean ± std as text above each boxplot
        x = positions[i]
        y = box['medians'][0].get_ydata()[0]
        mean = np.mean(category_grouped['W_new'])
        std = np.std(category_grouped['W_new'])
        text = f'{mean:.2f} ± {std:.2f}'
        axs[0, 0].text(x, y + 0.2, text, ha='center', va='bottom', fontsize=10)

    # Set the font size of y and x labels and ticks
    axs[0, 0].set_ylabel('Synaptic contacts', fontsize=12)
    axs[0, 0].set_xlabel(dataset_name, fontsize=12)
    axs[0, 0].tick_params(axis='both', which='both', labelsize=10)
    # Remove the background grid
    axs[0, 0].grid(False)
    # Remove the left and upper border lines
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    axs[0, 0].set_xticks(positions)
    axs[0, 0].set_xticklabels(categories)


    ################## Data next axis : Number of presynaptic partners
    # Iterate over unique categories in 'dorso-ventral'
    for i, category in enumerate(categories):
        # Filter the data for the current category
        category_data = syn_df[syn_df[category_column] == category]
        
        # Compute grouped data for the current category
        category_grouped = category_data.groupby(['instance_post']).agg({'type_pre': 'count'})
        
        # Plot the boxplot for the current category
        box = axs[0,1].boxplot(category_grouped['type_pre'], positions=[positions[i]], widths=0.4, showmeans=True, patch_artist= True)
        # Customize the box color
        for patch in box['boxes']:
            patch.set_facecolor(color_palette[i])
        
        # Customize the outlier markers
        for patch in box['fliers']:
            patch.set(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3)  # Black-filled rhombus
        
        # Add mean ± std as text above each boxplot
        x = positions[i]
        y = box['medians'][0].get_ydata()[0]
        mean = np.mean(category_grouped['type_pre'])
        std = np.std(category_grouped['type_pre'])
        text = f'{mean:.2f} ± {std:.2f}'
        axs[0, 1].text(x, y + 0.2, text, ha='center', va='bottom', fontsize=10)

    # Set the font size of y and x labels and ticks
    axs[0, 1].set_ylabel('Presynaptic partners', fontsize=12)
    axs[0, 1].set_xlabel(dataset_name, fontsize=12)
    axs[0, 1].tick_params(axis='both', which='both', labelsize=10)
    # Remove the background grid
    axs[0, 1].grid(False)
    # Remove the left and upper border lines
    axs[0, 1].spines['right'].set_visible(False)
    axs[0, 1].spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    axs[0, 1].set_xticks(positions)
    axs[0, 1].set_xticklabels(categories)


    ############ Data next axis: Number of presynaptic cell types
    # Iterate over unique categories
    for i, category in enumerate(categories):
        # Filter the data for the current category
        category_data = syn_df[syn_df[category_column] == category]
        
        # Compute grouped data for the current category
        category_grouped = category_data.groupby(['instance_post']).agg({'type_pre': 'nunique'})
        
        # Plot the boxplot for the current category
        box = axs[1, 0].boxplot(category_grouped['type_pre'], positions=[positions[i]], widths=0.4, showmeans=True, patch_artist = True)
        # Customize the box color
        for patch in box['boxes']:
            patch.set_facecolor(color_palette[i])
        
        # Customize the outlier markers
        for patch in box['fliers']:
            patch.set(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3)  # Black-filled rhombus
        
        # Add mean ± std as text above each boxplot
        x = positions[i]
        y = box['medians'][0].get_ydata()[0]
        mean = np.mean(category_grouped['type_pre'])
        std = np.std(category_grouped['type_pre'])
        text = f'{mean:.2f} ± {std:.2f}'
        axs[1, 0].text(x, y + 0.2, text, ha='center', va='bottom', fontsize=10)

    # Set the font size of y and x labels and ticks
    axs[1, 0].set_ylabel('Presynaptic cell types', fontsize=12)
    axs[1, 0].set_xlabel(dataset_name, fontsize=12)
    axs[1, 0].tick_params(axis='both', which='both', labelsize=10)
    # Remove the background grid
    axs[1, 0].grid(False)
    # Remove the left and upper border lines
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].spines['top'].set_visible(False)
    # Set the x-axis tick positions and labels
    axs[1, 0].set_xticks(positions)
    axs[1, 0].set_xticklabels(categories)


    ############ Data next axis: Cosine similarity
    if not d_v_filter:
        # Iterate over unique categories
        for i, category in enumerate(categories):
            # Filter the data for the current category
            category_grouped = cosine_sim_summary_df[cosine_sim_summary_df[category_column] == category]
            
            # Plot the boxplot for the current category
            box = axs[1, 1].boxplot(category_grouped['cosine_sim'], positions=[positions[i]], widths=0.4, showmeans=True, patch_artist = True)
            # Customize the box color
            for patch in box['boxes']:
                patch.set_facecolor(color_palette[i])
            
            # Customize the outlier markers
            for patch in box['fliers']:
                patch.set(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3)  # Black-filled rhombus
            
            # Add mean ± std as text above each boxplot
            x = positions[i]
            y = box['medians'][0].get_ydata()[0]
            mean = np.mean(category_grouped['cosine_sim'])
            std = np.std(category_grouped['cosine_sim'])
            text = f'{mean:.2f} ± {std:.2f}'
            #axs[1, 1].text(x, y + 0.2, text, ha='center', va='bottom', fontsize=10)

        # Set the font size of y and x labels and ticks
        axs[1, 1].set_ylabel('Cosine similarity', fontsize=12)
        axs[1, 1].set_xlabel(dataset_name, fontsize=12)
        axs[1, 1].tick_params(axis='both', which='both', labelsize=10)
        # Remove the background grid
        axs[1, 1].grid(False)
        # Remove the left and upper border lines
        axs[1, 1].spines['right'].set_visible(False)
        axs[1, 1].spines['top'].set_visible(False)
        # Set the x-axis tick positions and labels
        axs[1, 1].set_xticks(positions)
        axs[1, 1].set_xticklabels(categories)

        if save_figures:
            # Quick plot saving
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Counts-and-similarity-in_{dataset_name}_{neuron_of_interest}_by_{category_column}_{relative_or_absolute}.pdf'
            fig.savefig(save_path+figure_title)
            print('FIGURE: Box-plots comparing categories')
        plt.close(fig)


    #%% 
    ########################################  SCATTER PLOTS - BY CATEGORY ######################################
    #############################################    CORRELATIONS   ############################################

    ## TOTAL NUMBER OF SYNAPTIC COUNTS VS PARTNERS

    # Data
    all_W_new = syn_df.groupby(['instance_post']).agg({'W_new': sum})['W_new'].tolist()
    all_type_pre = syn_df.groupby(['instance_post']).agg({'type_pre': 'count'})['type_pre'].tolist()
    categories = syn_df[category_column].unique()

    #Figure
    # Initialize the figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    fig.suptitle(f'Comparison between {category_column}, {dataset_name}')

    # Define a color palette
    color_palette = sns.color_palette(color_cat_set, len(categories))

    for i, category in enumerate(categories):
        # Filter the data for the current category
        category_data = syn_df[syn_df[category_column] == category]
        
        # Compute grouped data for the current category
        category_W_new = category_data.groupby(['instance_post']).agg({'W_new': sum})['W_new'].tolist()
        category_type_pre = category_data.groupby(['instance_post']).agg({'type_pre': 'count'})['type_pre'].tolist()
        
        # Scatter plot for axs[0] and axs[2]
        sns.scatterplot(x=category_type_pre, y=category_W_new, color=color_palette[i], ax=axs[i])

        # Fit linear regression line and plot the diagonal reference line
        x, y = np.array(category_type_pre), np.array(category_W_new)
        slope, intercept = np.polyfit(x, y, 1)
        axs[i].plot(x, slope * x + intercept, color='black', linestyle='--')
        #axs[i].plot(x, x, color='gray', linestyle='--')

        # Compute Pearson correlation coefficient and p-value
        r_value, p_value = pearsonr(x, y)
        
        # Print R-squared and p-value
        axs[i].text(0.1, 0.9, f'R-squared: {r_value**2:.2f}', transform=axs[i].transAxes, fontsize=10)
        axs[i].text(0.1, 0.85, f'p-value: {p_value:.4f}', transform=axs[i].transAxes, fontsize=10)

        #Titles subplot
        axs[i].set_xlabel('Presynaptic partners', fontsize = 10)
        axs[i].set_ylabel('Synaptic contacts', fontsize = 10)
        axs[i].set_title(f'Scatter Plot and Correlation (Category: {category})', fontsize = 12)
        #Removing plot spines
        sns.despine(left=False, bottom=False)


    # Scatter plot and linear regression for axs[2]
    x_all, y_all = np.array(all_type_pre), np.array(all_W_new)
    sns.scatterplot(x=x_all, y=y_all, color=neuron_color, ax=axs[2])

    slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
    axs[2].plot(x_all, slope_all * x_all + intercept_all, color='black', linestyle='--')
    #axs[2].plot(x_all, x_all, color='gray', linestyle='--')

    r_value_all, p_value_all = pearsonr(x_all, y_all)

    # Print R-squared and p-value for axs[2]
    axs[2].text(0.1, 0.9, f'R-squared: {r_value_all**2:.2f}', transform=axs[2].transAxes, fontsize=10)
    axs[2].text(0.1, 0.85, f'p-value: {p_value_all:.4f}', transform=axs[2].transAxes, fontsize=10)

    # Set labels and title for each subplot
    axs[2].set_xlabel('Presynaptic partners', fontsize = 10)
    axs[2].set_ylabel('Synaptic contacts', fontsize = 10)
    axs[2].set_title('Scatter Plot and Correlation (All data)', fontsize = 12)
    #Removing plot spines
    sns.despine(left=False, bottom=False)

    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Linear-correlations-in_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Linear correlations comparing categories')
    plt.close(fig)


    ############################################ SCATTER PLOT - ALL PARTNERS ####################################
    #############################################    CORRELATIONS   ############################################

    #Data
    if relative_or_absolute == 'relative-counts':
        curr_df = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
    elif relative_or_absolute == 'absolute-counts':
        curr_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"

    curr_df = curr_df.fillna(0).copy()

    # Calculate the number of columns in the DataFrame
    num_cols = curr_df.shape[1]

    # Get all unique combinations of column pairs
    comparisons = list(combinations(curr_df.columns, 2))

    # Determine the number of subplots needed
    num_subplots = sum(pearsonr(curr_df[x_col], curr_df[y_col])[1] < 0.05 for x_col, y_col in comparisons)

    # Define the number of rows and columns for subplots
    num_cols_plot = int(np.ceil(np.sqrt(num_subplots)))
    num_rows = (num_subplots + num_cols_plot - 1) // num_cols_plot

    # Initialize the figure and gridspec only with necessary subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols_plot, figsize=(15, 3*num_rows), 
                            gridspec_kw={'height_ratios': [1]*num_rows})
    fig.suptitle('Scatter Plots and Correlations')

    # Variable to keep track of the current subplot index
    subplot_index = 0

    # Iterate over all pairs of columns
    for i, (x_col, y_col) in enumerate(comparisons):
        # Calculate the Pearson correlation and p-value
        r_value, p_value = pearsonr(curr_df[x_col], curr_df[y_col])
        p_value_corrected = len(comparisons) * p_value

        # Only plot if the p-value is less than 0.05
        if p_value_corrected < 0.05:
            # Get the data for the current pair of columns
            x_data, y_data = curr_df[x_col], curr_df[y_col]

            # Scatter plot for the current pair
            ax = axs[subplot_index // num_cols_plot, subplot_index % num_cols_plot]
            sns.scatterplot(x=x_data, y=y_data, ax=ax, color=neuron_color)
            
            # Fit linear regression line and get correlation values
            slope, intercept = np.polyfit(x_data, y_data, 1)

            # Plot the fitted line
            ax.plot(x_data, slope * x_data + intercept, color='black', linestyle='--')

            # Add text for R-squared and p-value
            ax.text(0.1, 0.85, f'R-squared: {r_value:.2f}', transform=ax.transAxes, fontsize=10)
            ax.text(0.1, 0.7, f'p-value: {p_value_corrected:.4f}', transform=ax.transAxes, fontsize=10)

            # Set axis labels and title for the subplot
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')

            # Increment the subplot index
            subplot_index += 1

    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Linear-correlations-between-partners_{dataset_name}_{neuron_of_interest}_{relative_or_absolute}.pdf'
        if analyzing_cluster:
            figure_title = f'\Linear-correlations-between-partners_{dataset_name}_{neuron_of_interest}_{cluster_id}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Linar correlations comparing partners')
    plt.close(fig)
    #%% 
    ############################################### HEATMAP - PLOTS ############################################
    ############################################################################################################

    ################################################ BINARY PLOTS ##############################################
    ## Visualizing the existance of a presynaptic neuron
    #Heatmap plots

    #Data filtering
    _data =binary_df[presence_threshold_sorted_column_order].copy()

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
    _palette = sns.color_palette(hex_color, as_cmap=True)

    heatmap = sns.heatmap(cmap =_palette, data = _data, vmin=0, vmax=1, linewidths=0.2,
                    linecolor='k', cbar=False, ax = axs, square=True) 
    axs.set_title(f'{neuron_of_interest}, Binary: presence - absence, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Column')
    axs.set_xlabel('Presynaptic neuron')


    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "binary_df[presence_threshold_sorted_column_order]"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)

    if save_figures:
        # Quick plot saving
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Binary-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Binary heatmap plotted and saved')
    plt.close(fig)


    ################################################ INSTANCE COUNTS ##############################################
    ## Visualizing instance (copies of the same neuron type) counts
    #Heatmap plots

    #Specifi color settings for instances:

    color_palette_name = "tab10" # "magma", "rocket", "tab10", "plasma", , "viridis", "flare"

    #Data filtering
    curr_df = counting_instances_df.T[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"

    #Sorting based on max, sum and count
    column_order = curr_df.max().sort_values(ascending=False).index.tolist() # Sorting based on MAX of all values in the column
    curr_max_sorted_df = curr_df[column_order] # swapping order of columns
    column_order = curr_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
    curr_sum_sorted_df = curr_df[column_order] # swapping order of columns
    column_order = curr_df.count().sort_values(ascending=False).index.tolist() # Sorting based on COUNT of all values in the column
    curr_count_sorted_df = curr_df[column_order] # swapping order of columns

    #Sorting based on rank
    curr_rank_sorted_df = counting_instances_df.T[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    #curr_rank_sorted_df[presence_threshold_neuron_filter]


    # Data
    _data =curr_rank_sorted_df 

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
    max_count = int(max(counting_instances_df.max()))
    _palette = sns.color_palette(color_palette_name,max_count)

    #Plot
    heatmap = sns.heatmap(cmap=_palette, data=_data, vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, Instance count, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Columns')
    axs.set_xlabel('Presynaptic neurons')


    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Instance-count_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of instance counts plotted vertically and saved')
    plt.close(fig)



    #Figure
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
    max_count = int(max(counting_instances_df.max()))
    _palette = sns.color_palette(color_palette_name, max_count)

    # Plot (rotated 90 degrees)
    heatmap = sns.heatmap(cmap=_palette, data=_data.transpose(), vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, Instance count, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_xlabel('Columns')
    axs.set_ylabel('Presynaptic neurons')



    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Instance-count_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of instance counts plotted horizontally and saved')
    plt.close(fig)


    ################################################ RELATIVE  COUNTS  ##############################################
    # Visualization of presynaptic contact percentatge for all columns
    #Heatmap of presynaptic partners  colorcoded by relative synaptic count
    #Data
    _data = top_rank_popularity_rel_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    _vmin = min_desired_count
    _vmax= 48 # 48 is a common multiple of 3 and 4 #roundup(max(_data.max())) # rounding up to the next ten
    bin_width = 3# 5
    _palette = sns.color_palette("gist_ncar",n_colors=int(_vmax/bin_width)) # n_colors=int(roundup(max(_data.max()))/bin_width)

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))


    #First axis

    sns.heatmap(cmap = _palette, vmin=_vmin, vmax=_vmax, data = _data, cbar_kws={"shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, count %, (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_ylabel('Column')
    #axs.set_yticklabels(id_column)
    axs.set_xlabel('Presynaptic neuron')


    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)


    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)

    # Modify the legend ticks
    cbar = axs.collections[0].colorbar
    cbar.set_ticks(range(_vmin, _vmax + bin_width, bin_width))




    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Relative-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of relative counts plotted vertically and saved')
    plt.close(fig)


    #Figure
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))

    #First axis
    # Plot (rotated 90 degrees)
    sns.heatmap(cmap = _palette, vmin=_vmin, vmax=_vmax, data = _data.transpose(), cbar_kws={"shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, count %, (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_xlabel('Column')
    #axs.set_yticklabels(id_column)
    axs.set_ylabel('Presynaptic neuron')


    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)

    # Modify the legend ticks
    cbar = axs.collections[0].colorbar
    cbar.set_ticks(range(_vmin, _vmax + bin_width, bin_width))


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' ts'
        figure_title = f'\Relative-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of relative counts plotted horizontally and saved')
    plt.close(fig)



    ################################################### ABSOLUTE COUNTS  #################################################
    # Visualization of presynaptic contacts for all columns
    # Heatmap of presynaptic partners  colorcoded by absolute synaptic count


    #Data
    _data = top_rank_popularity_abs_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    _vmin = min_desired_count
    _vmax= 48 # 48 is a common multiple of 3 and 4 #roundup(max(_data.max())) # rounding up to the next ten
    bin_width = 3# 5
    _palette = sns.color_palette("gist_ncar",n_colors=int(_vmax/bin_width)) # 'rocket_r'

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))


    #First axis

    sns.heatmap(cmap = _palette, vmin=_vmin, vmax=_vmax, data = _data, cbar_kws={"shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, absolute count, (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_ylabel('Column')
    #axs.set_yticklabels(id_column)
    axs.set_xlabel('Presynaptic neuron')


    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)


    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)

    # Modify the legend ticks
    cbar = axs.collections[0].colorbar
    cbar.set_ticks(range(_vmin, _vmax + bin_width, bin_width))


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Absolute-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of absolute counts plotted vertically and saved')
    plt.close(fig)


    #Figure
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))

    #First axis
    # Plot (rotated 90 degrees)
    sns.heatmap(cmap = _palette, vmin=_vmin, vmax=_vmax, data = _data.transpose(), cbar_kws={"shrink": 0.25}, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, absolute count, (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_xlabel('Column')
    #axs.set_yticklabels(id_column)
    axs.set_ylabel('Presynaptic neuron')


    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)

    # Modify the legend ticks
    cbar = axs.collections[0].colorbar
    cbar.set_ticks(range(_vmin, _vmax + bin_width, bin_width))


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Absolute-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of absolute counts plotted horizontally and saved')
    plt.close(fig)


    if not d_v_filter:
        # Data sorted also by cosine similarity between columns and person correlation between presynaptic partners
        ordered_data = _data.iloc[cosine_row_order].copy()
        ordered_data = ordered_data.iloc[:,pearson_order].copy()

        #Figure
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))

        #First axis
        # Plot (rotated 90 degrees)
        sns.heatmap(cmap = _palette, vmin=_vmin, vmax=_vmax, data = ordered_data.transpose(), cbar_kws={"shrink": 0.25}, ax=axs, square=True)
        axs.set_title(f'{neuron_of_interest}, absolute count, (syn>={min_desired_count}), sorted by {sort_by}')
        axs.set_xlabel('Column (sorted by cosine similarity)')
        #axs.set_yticklabels(id_column)
        axs.set_ylabel('Presynaptic neuron (sorted by pearson correlation)')

        # Reducing font size of x-axis tick labels
        for tick_label in heatmap.get_xticklabels():
            tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

        # Reducing font size of y-axis tick labels
        for tick_label in heatmap.get_yticklabels():
            tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

        # Add ticks in the Y-axis for each row in "_data"
        axs.set_yticks(range(len(ordered_data.transpose().index)))
        axs.set_yticklabels(ordered_data.transpose().index)

        # Add ticks in the X-axis for each row in "_data"
        axs.set_xticks(range(len(ordered_data.transpose().columns)))
        axs.set_xticklabels(ordered_data.transpose().columns)

        # Modify the legend ticks
        cbar = axs.collections[0].colorbar
        cbar.set_ticks(range(_vmin, _vmax + bin_width, bin_width))


        #Plot saving
        if save_figures:
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Absolute-heatmap_{dataset_name}_{neuron_of_interest}_cosine-pearson.pdf'
            fig.savefig(save_path+figure_title)
            print('FIGURE: Visualization of absolute counts plotted horizontally and saved')
        plt.close(fig)



    ################################################### NEURONAL RANK  #################################################
    # Visualization neuron´s ranks
    # Heatmap of presynaptic partners colorcoded by rank


    #Data
    _data = rank_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    total_num_ranks = int(max(_data.max()))

    #Figure 
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
    #_palette = sns.color_palette("rocket",n_colors=20)
    _palette = sns.color_palette("tab20",n_colors=total_num_ranks)

    #First axis
    sns.heatmap(cmap = _palette, vmin=0, vmax=total_num_ranks, cbar_kws={"ticks":list(range(1,top_rank_df['rank'].max()+2,1)),"shrink": 0.25}, data = _data, ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, RANK neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_ylabel('Column')
    #axes.set_yticklabels(id_column)
    axs.set_xlabel('Presynaptic neuron')


    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)


    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Rank-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of ranks plotted vertically and saved')

    plt.close(fig)


    #Figure 
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
    #_palette = sns.color_palette("rocket",n_colors=20)
    _palette = sns.color_palette("tab20",n_colors=total_num_ranks)

    #First axis
    sns.heatmap(cmap = _palette, vmin=0, vmax=total_num_ranks, cbar_kws={"ticks":list(range(1,top_rank_df['rank'].max()+2,1)),"shrink": 0.25}, data = _data.transpose(), ax=axs, square=True)
    axs.set_title(f'{neuron_of_interest}, RANK neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs.set_xlabel('Column')
    #axes.set_yticklabels(id_column)
    axs.set_ylabel('Presynaptic neuron')


    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Rank-heatmap_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of ranks plotted horizontally and saved')

    plt.close(fig)


    #####################################        HISTOGRAMS PLOTS      ###############################################
    # Aim: evaluate if there are unexpected subpopulations

    _data = top_rank_popularity_abs_df[presence_threshold_sorted_column_order].copy()
    _data.fillna(0, inplace=True)

    # Define the number of subplots
    num_subplots = 3
    fig, axs = plt.subplots(num_subplots,2, figsize=(8, 6 * num_subplots))
    fig.suptitle('Count distribution')

    # Loop through the columns and create subplots
    for i in range(num_subplots):
        # Plot histogram
        sns.histplot(data=_data.iloc[:, i], kde=True, binwidth=3, ax=axs[i,0])
        axs[i,0].set_title('Absolute counts')

    _data = top_rank_popularity_rel_df[presence_threshold_sorted_column_order].copy()
    _data.fillna(0, inplace=True)

    # Loop through the columns and create subplots
    for i in range(num_subplots):
        # Plot histogram
        sns.histplot(data=_data.iloc[:, i], kde=True, binwidth=3, ax=axs[i,1])
        axs[i,1].set_title('Relative counts')


    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Count-histograms-main-inputs.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Counts histogram plotted and saved')

    ####################################### NEUROTRANSMITTER IDENTITY PLOTS ##########################################
    # Visualization of neurotranmitter identity for all columns
    # Heatmap of presynaptic partners  colorcoded by neurotransmitter (NT) type or polarity code

    ##########################################       IDENTITY      ###################################################
    # Data
    _data = syn_popularity_NT_type_df[presence_threshold_sorted_column_order].copy()
    NT_label_dict = {'None' : 1 ,'GLUT' : 2, 'GABA' : 3, 'ACH' : 4}

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
    _palette = sns.color_palette('Set2')
    palette = sns.color_palette([_palette[i] for i in NT_label_dict.values()])

    # Plot
    heatmap = sns.heatmap(data=_data, cmap=palette, cbar_kws={"shrink": 0.25}, ax=axs, square=True)

    # Set custom color bar labels
    ticks = [NT_label_dict[label] for label in NT_label_dict]
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(NT_label_dict.keys()))

    axs.set_title(f'{neuron_of_interest}, NT identity, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Columns')
    axs.set_xlabel('Presynaptic neurons')

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.8)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Identity_NT_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of neurotransmitter identiyt plotted vertically and saved')
    plt.close(fig)


    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(30*cm, 15*cm))
    _palette = sns.color_palette('Set2')
    palette = sns.color_palette([_palette[i] for i in NT_label_dict.values()])

    # Plot
    heatmap = sns.heatmap(data=_data.transpose(), cmap=palette, cbar_kws={"shrink": 0.25}, ax=axs, square=True)

    # Set custom color bar labels
    ticks = [NT_label_dict[label] for label in NT_label_dict]
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(NT_label_dict.keys()))

    axs.set_title(f'{neuron_of_interest}, NT identity, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Presynaptic neurons')
    axs.set_xlabel('Columns')

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Identity_NT_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of neurotransmitter identiyt plotted horizontally and saved')
    plt.close(fig)


    ##########################################      POLARITY     ###################################################
    # Data
    _data = syn_popularity_NT_polarity_df[presence_threshold_sorted_column_order].copy()
    NT_label_dict = {'Exc.' : 1 ,'Inh.':-1}

    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
    palette = ['red', 'blue']

    # Plot
    heatmap = sns.heatmap(data=_data, cmap=palette, cbar_kws={"shrink": 0.25}, ax=axs, square=True)

    # Set custom color bar labels
    ticks = [NT_label_dict[label] for label in NT_label_dict]
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(NT_label_dict.keys()))

    axs.set_title(f'{neuron_of_interest}, NT polarity, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Columns')
    axs.set_xlabel('Presynaptic neurons')

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.8)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.index)))
    axs.set_yticklabels(_data.index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.columns)))
    axs.set_xticklabels(_data.columns)


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Polarity_NT_{dataset_name}_{neuron_of_interest}_{sort_by}-vertical.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of neurotransmitter identiyt plotted vertically and saved')
    plt.close(fig)


    #Figure
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(30*cm, 15*cm))
    palette = ['red', 'blue']

    # Plot
    heatmap = sns.heatmap(data=_data.transpose(), cmap=palette, cbar_kws={"shrink": 0.25}, ax=axs, square=True)

    # Set custom color bar labels
    ticks = [NT_label_dict[label] for label in NT_label_dict]
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(NT_label_dict.keys()))

    axs.set_title(f'{neuron_of_interest}, NT polarity, syn>={min_desired_count}, sorted by {sort_by}')
    axs.set_ylabel('Presynaptic neurons')
    axs.set_xlabel('Columns')

    # Reducing font size of x-axis tick labels
    for tick_label in heatmap.get_xticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.2)

    # Reducing font size of y-axis tick labels
    for tick_label in heatmap.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

    # Add ticks in the Y-axis for each row in "_data"
    axs.set_yticks(range(len(_data.transpose().index)))
    axs.set_yticklabels(_data.transpose().index)

    # Add ticks in the X-axis for each row in "_data"
    axs.set_xticks(range(len(_data.transpose().columns)))
    axs.set_xticklabels(_data.transpose().columns)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Polarity_NT_{dataset_name}_{neuron_of_interest}_{sort_by}-horizontal.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of neurotransmitter identiyt plotted horizontally and saved')
    plt.close(fig)


    ################################################### COSINE SIMILARITY #################################################
    # Visualization cosine similatiry of column vectors (postsynaptic nueron´s input space)

    if not d_v_filter:
        #Data
        _data = _data_reordered_cosine_sim
        _data.dropna(how='all', inplace=True)    # now dropping if all values in the row are nan

        # Create a figure with custom grid layout
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(3, 2, width_ratios=[8, 1], height_ratios=[1.2, 8, 0.5])

        # Plot the dendrogram_cosine
        ax_dendrogram_cosine = plt.subplot(gs[0, :-1])
        ax_dendrogram_cosine.spines['top'].set_visible(False)
        ax_dendrogram_cosine.spines['right'].set_visible(False)
        ax_dendrogram_cosine.spines['bottom'].set_visible(False)
        ax_dendrogram_cosine.spines['left'].set_visible(False)
        ax_dendrogram_cosine.get_xaxis().set_visible(False)
        ax_dendrogram_cosine.get_yaxis().set_visible(False)
        hierarchy.dendrogram(dendrogram_cosine, ax=ax_dendrogram_cosine, color_threshold=0)

        # Plot the heatmap using the reordered DataFrame
        ax_heatmap = plt.subplot(gs[1, :-1])
        sns.heatmap(cosine_sim_reordered, cmap='coolwarm', annot=False, xticklabels=_data.index, yticklabels=_data.index, ax=ax_heatmap, cbar=False)
        #ax_heatmap.set_title('Cosine Similarity Heatmap')
        ax_heatmap.set_xlabel('Column')
        ax_heatmap.set_ylabel('Column')
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, fontsize=3)
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=3)

        # Create a dummy plot for the color bar
        dummy_cax = fig.add_subplot(gs[2, :-1])
        dummy_cax.set_xticks([])
        dummy_cax.set_yticks([])

        # Add color bar below the heatmap
        cbar = plt.colorbar(ax_heatmap.collections[0], cax=dummy_cax, orientation='horizontal')
        cbar.set_label('Cosine Similarity')

        #Plot saving
        if save_figures:
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Cosine-similarity_{dataset_name}_{neuron_of_interest}_{relative_or_absolute}.pdf'
            fig.savefig(save_path+figure_title)
            print('FIGURE: Visualization of cosine similarity, clustered')
        plt.close(fig)


    #%% 
    ############################################### STACKED BAR - PLOTS ############################################
    ################################################################################################################
    ################################# Binary, absolute counts and rank together ####################################
    #Seb coding here

    #Figure
    fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (40*cm, 15*cm))
    fig.tight_layout(pad=10) # Adding some space between subplots


    # First axis: Absolute counts

    # Data
    _data = top_rank_popularity_abs_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    # Binning the data
    bin_width = 3
    binned_data = np.floor(_data / bin_width) * bin_width
    # Setting limits and colors
    _vmin = min_desired_count
    _vmax= 48 # 48 is a common multiple of 3 and 4 #roundup(max(_data.max())) # rounding up to the next ten
    bin_width = 3# 5
    _palette = sns.color_palette("gist_ncar",n_colors=int(_vmax/bin_width))

    # Get unique values across all columns
    unique_values = np.unique(binned_data.T.values)
    # Determine the colors for each unique value
    colors = _palette[:len(unique_values)]
    # Calculate the value counts for each column
    value_counts = binned_data.apply(lambda x: x.value_counts())
    value_counts_norm = value_counts.apply(lambda x: (x / x.sum()), axis=0)
    # Plot the stacked bar chart
    value_counts_norm.T.plot(kind='bar', stacked=True, color=colors, legend=False, ax = axs[0])
    # Set the x-axis and y-axis labels
    axs[0].set_xlabel('Presynaptic neuron')
    axs[0].set_ylabel('Absolute synapse number (% of counts)')
    # Set the x-axis tick labels
    axs[0].set_xticklabels(value_counts.T.index, rotation=90)
    # Set the plot title
    axs[0].set_title(f'{neuron_of_interest}, absolute-count counts, (syn>={min_desired_count}), sorted by {sort_by}')
    # Remove spines
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)

    # Create legend patches with bin width labels 
    # legend_patches = [
    #     mpatches.Patch(facecolor=color, label=f'{int(value)}-{int(value+bin_width)}' if not np.isnan(value) else '')
    #     for color, value in zip(colors, unique_values)
    # ]
    # # Add the modified legend, excluding blanks from the labels
    # axs[0].legend(handles=legend_patches, title='Binned Values', bbox_to_anchor=(1.05, 1)) #  loc='upper right'

    # Next axis: Ranks
    # Data
    _data = rank_df[presence_threshold_sorted_column_order].copy()#  filtering based on " presence_threshold"
    _palette = sns.color_palette("tab20",n_colors=total_num_ranks)
    # Determine the colors for each unique value
    colors = _palette[:len(unique_values)]
    # Calculate the value counts for each column
    value_counts = _data.apply(lambda x: x.value_counts())
    value_counts_norm = value_counts.apply(lambda x: (x / x.sum()), axis=0)
    # Plot the stacked bar chart
    value_counts_norm.T.plot(kind='bar', stacked=True, color=colors, legend=False, ax = axs[1])
    # Set the x-axis and y-axis labels
    axs[1].set_xlabel('Presynaptic neuron')
    axs[1].set_ylabel('Rank (% of counts)')
    # Set the x-axis tick labels
    axs[1].set_xticklabels(value_counts.T.index, rotation=90)
    # Set the plot title
    axs[1].set_title(f'{neuron_of_interest}, Rank counts, (syn>={min_desired_count}), sorted by {sort_by}')
    # Remove spines
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)


    # Next axis: binary data
    sorted_thr_rel_presence_absence_df['Presynaptic neuron_cat'] = pd.Categorical(
        sorted_thr_rel_presence_absence_df['Presynaptic neuron'], 
        categories=presence_threshold_sorted_column_order, 
        ordered=True
    )
    sorted_thr_rel_presence_absence_df.sort_values('Presynaptic neuron_cat', inplace = True)
    sorted_thr_rel_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                        edgecolor = "black", ax = axs[2],legend=False)
    axs[2].set_title(f'{neuron_of_interest}, Presence above {int(presence_threshold*100)}%, syn>={min_desired_count}, sorted by {sort_by}')
    #axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[2].set_xlabel('Presynaptic neuron')
    axs[2].set_ylabel('Presence (% of columns)')
    # Remove spines
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)



    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Stacked-bar-plots_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of absolute counts plotted as stacked bars and saved')
    plt.close(fig)



    ################################################# NEURONAL RANK  ##############################################
    #TODO Stacked plots 


    ############################################ CORRELATION MATRIXES  #############################################
    # Visualization of Hieracrchical clustering
    # Heatmap of presynaptic partners' person correlation
    # Relative numbers
    if relative_or_absolute == 'relative-counts':
        correlation_rel_no_NaN_df.replace(np.NaN,0.0, inplace = True) # Relevant for the dendogram not to have NaNs (Set to 0.0 for color map purposes)
        _palette = sns.color_palette("vlag", as_cmap=True) # Diverging palette



        #Plot with dendogram
        if permutation_analysis:
            g = sns.clustermap(cmap = _palette, data = correlation_rel_no_NaN_df, annot=p_values_correlation_rel_after_permutation_df, fmt='', annot_kws={"size":8, "color": "k"})
            g.fig.suptitle(f'{neuron_of_interest} partners, p-values from correlations that passed permutation test, relative count(syn>={min_desired_count})') 
            #g = sns.clustermap(cmap = _palette, data = correlation_rel_no_NaN_df, annot=p_value_permutation_rel_df .applymap(filter_values), fmt='', annot_kws={"size":8, "color": "k"})
        else:
            g = sns.clustermap(cmap = _palette, data = correlation_rel_no_NaN_df, annot = np.array(p_values_correlation_rel_no_NaN_df_asterix_df), fmt='', annot_kws={"size":16, "color": "k"})
            g.fig.suptitle(f'{neuron_of_interest} partners, p-values from correlations, relative count(syn>={min_desired_count})') 


        g.ax_heatmap.set_xlabel('Presynaptic neuron',fontsize = 14)
        g.ax_heatmap.set_ylabel('Presynaptic neuron',fontsize = 14)
        g.fig.subplots_adjust(top=0.9)
        x0, y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0+1, y0, g.ax_cbar.get_position().width/5, g.ax_cbar.get_position().width])
        g.ax_cbar.set_title('pearson')



        #Plot saving
        if save_figures:
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Hierarchical-clustering-correlation-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
            g.savefig(save_path+figure_title)
            print('FIGURE: Visualization of pearson correlation and hierarchical clustering plotted and saved')
        plt.close(g.fig)


        #Plot same without dendogram
        #TODO Do it the way Buraks did it for the paper





    elif relative_or_absolute == 'absolute-counts':
        # Absolute numbers
        correlation_abs_no_NaN_df.replace(np.NaN,0.0, inplace = True) # Relevant for the dendogram not to have NaNs (Set to 0.0 for color map purposes)
        _palette = sns.color_palette("vlag", as_cmap=True) # Diverging palette
        if permutation_analysis:
            g = sns.clustermap(cmap = _palette, data = correlation_abs_no_NaN_df, annot=p_values_correlation_abs_after_permutation_df, fmt='', annot_kws={"size":8, "color": "k"})
            g.fig.suptitle(f'{neuron_of_interest} partners, p-values from correlations that passed permutation test, absolute count(syn>={min_desired_count})') 
            #g = sns.clustermap(cmap = _palette, data = correlation_abs_no_NaN_df, annot=p_value_permutation_abs_df .applymap(filter_values), fmt='', annot_kws={"size":8, "color": "k"})
        else:
            g = sns.clustermap(cmap = _palette, data = correlation_abs_no_NaN_df, annot = np.array(p_values_correlation_abs_no_NaN_df_asterix_df), fmt='', annot_kws={"size":16, "color": "k"})
            g.fig.suptitle(f'{neuron_of_interest} partners, p-values from correlations, absolute count(syn>={min_desired_count})') 


        g.ax_heatmap.set_xlabel('Presynaptic neuron')
        g.ax_heatmap.set_ylabel('Presynaptic neuron')
        g.fig.subplots_adjust(top=0.9)
        x0, y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0+1, y0, g.ax_cbar.get_position().width/5, g.ax_cbar.get_position().width])
        g.ax_cbar.set_title('pearson')


        #Plot saving
        if save_figures:
            save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Hierarchical-clustering-correlation-absolute-counts_{dataset_name}_{neuron_of_interest}.pdf'
            if analyzing_cluster:
                figure_title = f'\Hierarchical-clustering-correlation-absolute-counts_{dataset_name}_{neuron_of_interest}_{cluster_id}.pdf'
            g.savefig(save_path+figure_title)
            print('FIGURE: Visualization of pearson correlation and hierarchical clustering plotted and saved')
        plt.close(g.fig)

    ################################ PERMUTATION PLOTS FOR THE CORRELATION MATRIX ###############################

    if permutation_analysis:
        if relative_or_absolute == 'absolute-counts':
            ### Absolute counts
            #Plotting p-values <0.05 from the permutation
            fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (40*cm, 15*cm))

            # Create the heatmap and annotate cells with p-values < 0.05
            sns.heatmap(p_value_permutation_abs_df , annot=p_value_permutation_abs_df .applymap(filter_values), cmap='coolwarm', center=0.05, fmt='', ax = axs[0])
            axs[0].set_title('Permutation test - p-values')

            sns.heatmap(observed_corr_abs_df, cmap='coolwarm', ax = axs[1])
            axs[1].set_title('Observed correlation')

            if save_figures:
                save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
                figure_title = f'\Permutation-test-pvalues_{dataset_name}_{neuron_of_interest}_{relative_or_absolute}.pdf'
                fig.savefig(save_path+figure_title)
                print('FIGURE: Visualization of permutation test values of pearson correlations')
            plt.close(fig)
        
        elif relative_or_absolute == 'relative-counts':
            ### Relative counts
            #Plotting p-values <0.05 from the permutation
            fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (40*cm, 15*cm))

            # Create the heatmap and annotate cells with p-values < 0.05
            sns.heatmap(p_value_permutation_rel_df , annot=p_value_permutation_rel_df .applymap(filter_values), cmap='coolwarm', center=0.05, fmt='', ax = axs[0])
            axs[0].set_title('Permutation test - p-values')

            sns.heatmap(observed_corr_rel_df, cmap='coolwarm', ax = axs[1])
            axs[1].set_title('Observed correlation')

            if save_figures:
                save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
                figure_title = f'\Permutation-test-pvalues_{dataset_name}_{neuron_of_interest}_{relative_or_absolute}.pdf'
                fig.savefig(save_path+figure_title)
                print('FIGURE: Visualization of permutation test values of pearson correlations')
            plt.close(fig)

    ######################################### DISTRIBUTIONS FROM PERMUTATIONS ########################################
    ### Plotting all distributions  of correlations values during the permutation per each pair

    if permutation_analysis:
        if relative_or_absolute == 'absolute-counts':
            ### Absolute counts
            # Data
            ### Absolute counts
            curr_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy()
            curr_df = curr_df.fillna(0).copy()
            if analyzing_cluster:
                curr_dataset_abs_df = dataset_abs_df.fillna(0).copy()
            else:
                curr_dataset_abs_df = curr_df.copy() 

            # Columns to be compared
            column_names = curr_df.columns.tolist()

            # In case you want to run it for different subsets of the data, code needs to be modified for defining column_names
            clusters_list = [curr_df]  # Insert your actual cluster DataFrames

            # Create subplots to visualize the results
            valid_pairs = list(combinations(column_names, 2))

            # Define the number of subplots per figure
            subplots_per_figure = 16
            num_figures = int(np.ceil(len(valid_pairs) / subplots_per_figure))

            # Create and save figures with subplots
            figure_title = f'\Correlation_permutation_plots_{dataset_name}_{neuron_of_interest}_{cluster_id}_{relative_or_absolute}.pdf'
            outputPath =  save_path + figure_title
            with PdfPages(outputPath) as pdf:
                for fig_num in range(num_figures):
                    start_idx = fig_num * subplots_per_figure
                    end_idx = min((fig_num + 1) * subplots_per_figure, len(valid_pairs))

                    num_rows = num_cols = int(np.ceil(np.sqrt(end_idx - start_idx)))
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

                    for pair_idx, (column1_name, column2_name) in enumerate(valid_pairs[start_idx:end_idx]):
                        observed_corr, p_value, shuffled_corrs = permutation_test(curr_df, curr_dataset_abs_df, column1_name, column2_name, num_permutations, _seed)
                        ax = axes[pair_idx // num_cols, pair_idx % num_cols]

                        #In case it was not possible to get data for shuffled_corrs 
                        if np.all(np.isnan(shuffled_corrs)):
                            ax.set_title(f"{column1_name} vs. {column2_name}")
                            ax.text(0.1, 0.5, "Permutation test was not possible", fontsize=10, fontweight='bold')
                        else:
                            # Plot the observed correlation and the distribution of shuffled correlations in the corresponding subplot
                            ax.hist(shuffled_corrs, bins=30, alpha=0.6, color='gray', edgecolor='black', label='Shuffled')
                            ax.axvline(observed_corr, color='red', linestyle='dashed', linewidth=2, label='Observed')
                            ax.set_title(f"{column1_name} vs. {column2_name}")
                            ax.legend()

                            # Annotate the p-value in the plot
                            ax.text(0.8, 0.85, f"P-value: {p_value:.3f}", transform=ax.transAxes, fontsize=10, fontweight='bold')

                    # Hide empty subplots if any
                    for pair_idx in range(end_idx - start_idx, num_rows * num_cols):
                        axes[pair_idx // num_cols, pair_idx % num_cols].axis('off')

                    if save_figures:
                        pdf.savefig(fig)
                    plt.close(fig)

        elif relative_or_absolute == 'relative-counts':
            ### Reletive counts
            # Data
            ### Absolute counts
            curr_df = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy()
            curr_df = curr_df.fillna(0).copy()
            if analyzing_cluster:
                curr_dataset_rel_df = dataset_rel_df.fillna(0).copy()
            else:
                curr_dataset_rel_df = curr_df.copy() 

            # Columns to be compared
            column_names = curr_df.columns.tolist()

            # In case you want to run it for different subsets of the data, code needs to be modified for defining column_names
            clusters_list = [curr_df]  # Insert your actual cluster DataFrames

            # Create subplots to visualize the results
            valid_pairs = list(combinations(column_names, 2))

            # Define the number of subplots per figure
            subplots_per_figure = 16
            num_figures = int(np.ceil(len(valid_pairs) / subplots_per_figure))

            # Create and save figures with subplots
            figure_title = f'\Correlation_permutation_plots_{dataset_name}_{neuron_of_interest}_{cluster_id}_{relative_or_absolute}.pdf'
            outputPath =  save_path + figure_title
            with PdfPages(outputPath) as pdf:
                for fig_num in range(num_figures):
                    start_idx = fig_num * subplots_per_figure
                    end_idx = min((fig_num + 1) * subplots_per_figure, len(valid_pairs))

                    num_rows = num_cols = int(np.ceil(np.sqrt(end_idx - start_idx)))
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

                    for pair_idx, (column1_name, column2_name) in enumerate(valid_pairs[start_idx:end_idx]):
                        observed_corr, p_value, shuffled_corrs = permutation_test(curr_df, curr_dataset_rel_df, column1_name, column2_name, num_permutations, _seed)
                        ax = axes[pair_idx // num_cols, pair_idx % num_cols]

                        #In case it was not possible to get data for shuffled_corrs 
                        if np.all(np.isnan(shuffled_corrs)):
                            ax.set_title(f"{column1_name} vs. {column2_name}")
                            ax.text(0.1, 0.5, "Permutation test was not possible", fontsize=10, fontweight='bold')
                        else:
                            # Plot the observed correlation and the distribution of shuffled correlations in the corresponding subplot
                            ax.hist(shuffled_corrs, bins=30, alpha=0.6, color='gray', edgecolor='black', label='Shuffled')
                            ax.axvline(observed_corr, color='red', linestyle='dashed', linewidth=2, label='Observed')
                            ax.set_title(f"{column1_name} vs. {column2_name}")
                            ax.legend()

                            # Annotate the p-value in the plot
                            ax.text(0.8, 0.85, f"P-value: {p_value:.3f}", transform=ax.transAxes, fontsize=10, fontweight='bold')

                    # Hide empty subplots if any
                    for pair_idx in range(end_idx - start_idx, num_rows * num_cols):
                        axes[pair_idx // num_cols, pair_idx % num_cols].axis('off')

                    if save_figures:
                        pdf.savefig(fig)
                    plt.close(fig)


    ############################################     BOX-PLOTS     ##############################################
    #############################################################################################################

    ############################################# PRESYNAPTIC COUNTS ############################################
    # Relative and absolute counts across columns

    # Data
    _data = syn_popularity_rel_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    #Figure
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(30*cm, 30*cm))
    fig.tight_layout(pad=10) # Adding some space between subplots


    # First axes 

    sns.boxplot(data = _data, ax = axs[0])  # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[0].set_title(f'{neuron_of_interest},  count % of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[0].set_ylabel('Synaptic count (%) ', size = 12)
    axs[0].set_xlabel('Presynaptic neuron', size = 12)
    axs[0].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[0].set_yticklabels(axs[0].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)


    # Data
    _data = syn_popularity_abs_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    # Next axes 
    sns.boxplot(data = _data, ax = axs[1]) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[1].set_title(f'{neuron_of_interest},  Absolute count of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[1].set_ylabel('Synaptic count', size = 12)
    axs[1].set_xlabel('Presynaptic neuron', size = 12)
    axs[1].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[1].set_yticklabels(axs[1].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)


    #Plot saving
    if save_figures:
        save_path =  f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Box-plot-presynaptic-partners_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of presynaptic partners contacts')
    plt.close(fig)


    ###################################### COSINE SIMILARITY - CLUSTERS #########################################
    # Cosine values within a cluster and relative to other clusters

    if all([cluster_with_dendrogram, not d_v_filter]):
        ### Within a cluster
        # Data preprocessing 
        df_cluster = pd.DataFrame({cluster_name: pd.Series(cluster_values.flatten()) for cluster_name, cluster_values in cosine_cluster_arrays.items()})
        df_cluster = df_cluster.round(2).copy()
        df_cluster[df_cluster == 1.00] = np.nan

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10*cm, 20*cm))

        # Create a color palette for boxplot colors
        color_palette = sns.color_palette("Set3", n_colors=len(cosine_cluster_arrays))

        # First axis
        # Create the boxplot
        sns.boxplot(data=df_cluster, palette=color_palette, ax=axs[0])
        # Set labels and title
        axs[0].set_xlabel('Clusters')
        axs[0].set_ylabel('Cosine similarity')
        axs[0].set_title('Within cluster similarity')
        axs[0].grid(False)
        # Remove the left and upper border lines
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)

        # Second axis
        # Create the boxplot using pandas
        box = df_cluster.boxplot(patch_artist=True, ax=axs[1])

        # Set labels and title
        axs[1].set_xlabel('Clusters')
        axs[1].set_ylabel('Cosine similarity')
        axs[1].set_title('Number of columns in each cluster')

        # Call the function to add N labels
        add_n_labels(axs[1], cosine_cluster_arrays,df_cluster)


        #Plot saving
        if save_figures:
            save_path =  f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
            figure_title = f'\Box-plot-within-cluster-similarity_{dataset_name}_{neuron_of_interest}_cosine_sim.pdf'
            fig.savefig(save_path+figure_title)
            print('FIGURE: Visualization of cosine similarity within cluster')
        plt.close(fig)



        #TODO
        ### Between clusters



    #############################################    VIOLIN-PLOTS     ##############################################
    ################################################################################################################

    ############################################# PRESYNAPTIC COUNTS ###############################################

    # Data
    if relative_or_absolute == 'absolute-counts':
        _data = top_rank_popularity_abs_df[presence_threshold_sorted_column_order].copy()
    elif relative_or_absolute == 'relative-counts':
        _data = top_rank_popularity_rel_df[presence_threshold_sorted_column_order].copy()

        dropped_indexes = []
        kept_indexes = []
        dropped_data = _data.dropna(how='all', inplace=False)
        dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
        kept_indexes.extend(dropped_data.index)



    # Relative and absolute counts across columns
    # Data
    _data = syn_popularity_rel_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    dropped_indexes = []
    kept_indexes = []
    dropped_data = _data.dropna(how='all', inplace=False)
    dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
    kept_indexes.extend(dropped_data.index)


    data_zeros = _data.fillna(0)
    syn_df_grouped = syn_df.groupby(['instance_post','dorso-ventral','hemisphere']).agg({'W_new': sum}).loc[kept_indexes]
    syn_df_grouped.reset_index(level='dorso-ventral', inplace= True)
    syn_df_grouped.reset_index(level='hemisphere', inplace= True)
    data_zeros['hemisphere'] = syn_df_grouped['hemisphere']
    data_zeros = data_zeros.reset_index()



    #Figure
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(30*cm, 30*cm))
    fig.tight_layout(pad=10) # Adding some space between subplots

    # First axes 
    sns.violinplot(data = data_zeros, ax = axs[0], scale='width')
    axs[0].set_title(f'{neuron_of_interest},  count % of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[0].set_ylabel('Synaptic count (%) ', size = 12)
    axs[0].set_xlabel('Presynaptic neuron', size = 12)
    axs[0].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[0].set_ylim(top = 65) #set same y lim for Tm9 and Tm1
    axs[0].set_yticklabels(axs[0].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)


    # Data
    _data = syn_popularity_abs_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    dropped_indexes = []
    kept_indexes = []
    dropped_data = _data.dropna(how='all', inplace=False)
    dropped_indexes.extend(list(set(_data.index) - set(dropped_data.index)))
    kept_indexes.extend(dropped_data.index)


    data_zeros = _data.fillna(0)
    syn_df_grouped = syn_df.groupby(['instance_post','dorso-ventral','hemisphere']).agg({'W_new': sum}).loc[kept_indexes]
    syn_df_grouped.reset_index(level='dorso-ventral', inplace= True)
    syn_df_grouped.reset_index(level='hemisphere', inplace= True)
    data_zeros['hemisphere'] = syn_df_grouped['hemisphere']
    data_zeros = data_zeros.reset_index()

    # Next axes 
    #sns.boxplot(data = data_zeros, ax = axs[1]) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    sns.violinplot(data = data_zeros, ax = axs[1], scale='width')
    axs[1].set_title(f'{neuron_of_interest},  Absolute count of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[1].set_ylabel('Synaptic count', size = 12)
    axs[1].set_xlabel('Presynaptic neuron', size = 12)
    axs[1].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    #axs[1].set_ylim(top = 60)
    axs[1].set_yticklabels(axs[1].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)


    #Plot saving
    if save_figures:
        save_path =  f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Violin-plot-presynaptic-partners_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of presynaptic partners contacts')
    plt.close(fig)

    
    #############################################     SWARM-PLOTS     ##############################################
    ################################################################################################################

    ############################################# PRESYNAPTIC COUNTS ###############################################
    _bound = 0.3 # 30% above or below the mean
    hline_span = 0.3 # how long the lines for the bounds are

    # Relative and absolute counts across columns
    # Data
    _data = syn_popularity_rel_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    # Calculate mean and upper/lower bounds
    mean_values = _data.mean(axis=0)
    upper_bound = mean_values + _bound * mean_values
    lower_bound = mean_values - _bound * mean_values



    #Figure
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(30*cm, 30*cm))
    fig.tight_layout(pad=10) # Adding some space between subplots

    # First axes 
    sns.swarmplot(data = _data, ax = axs[0])
    axs[0].set_title(f'{neuron_of_interest},  count % of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[0].set_ylabel('Synaptic count (%) ', size = 12)
    axs[0].set_xlabel('Presynaptic neuron', size = 12)
    axs[0].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[0].set_ylim(top = 65) #set same y lim for Tm9 and Tm1
    axs[0].set_yticklabels(axs[0].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)

    # Plot upper and lower bounds lines within the x-axis limits
    # Extract x-axis tick positions
    x_positions = axs[0].get_xticks()
    # Plot upper and lower bounds lines at tick positions for the first subplot
    for col, upper, lower, x, mean in zip(_data.columns, upper_bound, lower_bound, x_positions, mean_values):
        # Calculate custom xmin and xmax values
        xmin = x - hline_span
        xmax = x + hline_span
        # Create a horizontal line using ax.plot
        axs[0].plot([xmin, xmax], [upper, upper], color='k', linestyle='--',zorder=3)
        axs[0].plot([xmin, xmax], [lower, lower], color='k', linestyle='--',zorder=3)
        axs[0].plot([xmin, xmax], [mean, mean], color='g', linestyle='-',zorder=3)

    # Data
    _data = syn_popularity_abs_df.copy()[presence_threshold_sorted_column_order]#  filtering based on " presence_threshold"

    # Calculate mean and upper/lower bounds
    mean_values = _data.mean(axis=0)
    upper_bound = mean_values + _bound * mean_values
    lower_bound = mean_values - _bound * mean_values

    # Next axes 
    sns.swarmplot(data = _data, ax = axs[1])
    axs[1].set_title(f'{neuron_of_interest},  Absolute count of popular neurons (syn>={min_desired_count}), sorted by {sort_by}')
    axs[1].set_ylabel('Synaptic count', size = 12)
    axs[1].set_xlabel('Presynaptic neuron', size = 12)
    axs[1].set_xticklabels(_data, rotation=90, size = 10) # old sorting: _data[_data.max().sort_values(ascending = False).index]
    axs[1].set_yticklabels(axs[1].get_yticks(), size = 8)
    sns.despine(left=False, bottom=False)

    # Plot upper and lower bounds lines within the x-axis limits
    # Extract x-axis tick positions
    x_positions = axs[1].get_xticks()
    # Plot upper and lower bounds lines at tick positions for the first subplot
    for col, upper, lower, x, mean in zip(_data.columns, upper_bound, lower_bound, x_positions, mean_values):
        # Calculate custom xmin and xmax values
        xmin = x - hline_span
        xmax = x + hline_span
        # Create a horizontal line using ax.plot
        axs[1].plot([xmin, xmax], [upper, upper], color='k', linestyle='--',zorder=3)
        axs[1].plot([xmin, xmax], [lower, lower], color='k', linestyle='--',zorder=3)
        axs[1].plot([xmin, xmax], [mean, mean], color='g', linestyle='-',zorder=3)

    #Plot saving
    if save_figures:
        save_path =  f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\Swarm-plot-presynaptic-partners_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of presynaptic partners contacts')
    plt.close(fig)

    ################################################ BAR - PLOTS ################################################
    #############################################################################################################

    # Plotting bar plots of presynaptic counts
    # Quick plot across neurons of basic descriptive statistics of variability

    #Figure
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(30*cm, 15*cm))
    fig.tight_layout(pad=8) # Adding some space between subplots

    #Data
    if exclude_outliers:
        _data = curr_rel_stats_no_ouliers_df.round(2).copy()[presence_threshold_sorted_column_order+['mean']]#  filtering based on " presence_threshold"
        figure_title = f'\Variability-measures_{dataset_name}_{neuron_of_interest}_{sort_by}_no_outliers.pdf'
        axs[0,0].set_title(f'{neuron_of_interest} partners, variability measure: std, % of count(syn>={min_desired_count}), sorted by {sort_by}, no outliers')
        axs[0,1].set_title(f'{neuron_of_interest} partners, variability measure: C.V, % of count(syn>={min_desired_count}), sorted by {sort_by}, no outliers')
    else: 
        _data = curr_rel_stats_df.round(2).copy()[presence_threshold_sorted_column_order+['mean']]#  filtering based on " presence_threshold"
        figure_title = f'\Variability-measures_{dataset_name}_{neuron_of_interest}_{sort_by}.pdf'
        axs[0,0].set_title(f'{neuron_of_interest} partners, variability measure: std, % of count(syn>={min_desired_count}), sorted by {sort_by}')
        axs[0,1].set_title(f'{neuron_of_interest} partners, variability measure: C.V, % of count(syn>={min_desired_count}), sorted by {sort_by}')

    #First axis
    sns.barplot(data = _data.iloc[[2]], ax = axs[0,0] )
    axs[0,0].axhline(y = _data.iloc[[2]]['mean'][0], color = 'k', linestyle = 'dashed')  
    axs[0,0].set_ylabel(_data.index[2], size = 10)
    axs[0,0].set_xlabel(f'Presynaptic neuron', size = 10)
    axs[0,0].set_xticklabels(_data.iloc[[2]], rotation=90)
    #axs[0,0].set_yticklabels(axs[0,0].get_yticks().round(1), size = 6)
    axs[0,0].set_xticklabels(_data.columns.tolist(), size = 8)
    axs[0,0].set_ylim(0,8)
    sns.despine(left=False, bottom=False)


    #Next axis
    sns.barplot(data = _data.iloc[[-1]], ax = axs[0,1] )
    axs[0,1].axhline(y = _data.iloc[[-1]]['mean'][0], color = 'k', linestyle = 'dashed')  
    axs[0,1].set_ylabel(_data.index[-1], size = 10)
    axs[0,1].set_xlabel(f'Presynaptic neuron', size = 10)
    axs[0,1].set_xticklabels(_data.iloc[[-1]], rotation=90)
    #axs[0,1].set_yticklabels(axs[0,1].get_yticks().round(1), size = 6)
    axs[0,1].set_xticklabels(_data.columns.tolist(), size = 8)
    axs[0,1].set_ylim(0,0.6)
    sns.despine(left=False, bottom=False)


    #Data
    if exclude_outliers:
        _data = curr_abs_stats_no_ouliers_df.round(2).copy()[presence_threshold_sorted_column_order+['mean']]#  filtering based on " presence_threshold"
        axs[1,0].set_title(f'{neuron_of_interest} partners, variability measure: std, absolute count(syn>={min_desired_count}), sorted by {sort_by}, no outliers')
        axs[1,1].set_title(f'{neuron_of_interest} partners, variability measure: C.V, absolute count(syn>={min_desired_count}), sorted by {sort_by}, no outliers')
    else: 
        _data = curr_abs_stats_df.round(2).copy()[presence_threshold_sorted_column_order+['mean']]#  filtering based on " presence_threshold"
        axs[1,0].set_title(f'{neuron_of_interest} partners, variability measure: std, absolute count(syn>={min_desired_count}), sorted by {sort_by}')
        axs[1,1].set_title(f'{neuron_of_interest} partners, variability measure: C.V, absolute count(syn>={min_desired_count}), sorted by {sort_by}')

    #Next axis
    sns.barplot(data = _data.iloc[[2]], ax = axs[1,0] )
    axs[1,0].axhline(y = _data.iloc[[2]]['mean'][0], color = 'k', linestyle = 'dashed')  
    axs[1,0].set_ylabel(_data.index[2], size = 10)
    axs[1,0].set_xlabel(f'Presynaptic neuron', size = 10)
    axs[1,0].set_xticklabels(_data.iloc[[2]], rotation=90)
    #axs[1,0].set_yticklabels(axs[1,0].get_yticks().round(1), size = 6)
    axs[1,0].set_xticklabels(_data.columns.tolist(), size = 8)
    axs[1,0].set_ylim(0,45)
    sns.despine(left=False, bottom=False)

    #Next axis
    sns.barplot(data = _data.iloc[[-1]], ax = axs[1,1] )
    axs[1,1].axhline(y = _data.iloc[[-1]]['mean'][0], color = 'k', linestyle = 'dashed')  
    axs[1,1].set_ylabel(_data.index[-1], size = 10)
    axs[1,1].set_xlabel(f'Presynaptic neuron', size = 10)
    axs[1,1].set_xticklabels(_data.iloc[[-1]], rotation=90)
    #axs[1,1].set_yticklabels(axs[1,1].get_yticks().round(1), size = 6)
    axs[1,1].set_xticklabels(_data.columns.tolist(), size = 8)
    axs[1,1].set_ylim(0,0.6)
    sns.despine(left=False, bottom=False)


    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of variability measures')
    plt.close(fig)




    ################################################ HISTOGRAMS - PLOTS #########################################
    #############################################################################################################
    # Visualization across data sets


    # Plotting ExM and Em data together for following neurons
    partner_list = list(ExM_EM_absolut_counts.keys())
    binwidth_list = [2,1,4,2]

    # Determine the number of rows and columns for subplots
    num_rows = int((len(partner_list) + 1) / 2)
    num_cols = 2

    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    axs = axs.flatten()

    # Iterate over the list of partners
    for i, partner in enumerate(partner_list):
        _data = ExM_EM_absolut_counts[partner].copy()

        histplot = sns.histplot(
            data=_data, x='W_new', hue="fly_ID", multiple="dodge", shrink=1,
            log_scale=False, element="bars", fill=True, binwidth=binwidth_list[i],
            cumulative=False, stat="percent", common_norm=False, legend=False,
            ax=axs[i])

        axs[i].set_title(f'{partner}>{neuron_of_interest}, Column counts histogram ExM and EM, syn>={min_desired_count}')
        axs[i].set_ylabel('Column count (percent)', size=10)
        axs[i].set_xlabel('Presynaptic contacts', size=10)
        axs[i].tick_params(axis='x', labelsize=8)
        axs[i].tick_params(axis='y', labelsize=8)

        unique_labels = _data['fly_ID'].unique()
        handles = [plt.Rectangle((0, 0), 1, 1, color=sns.color_palette()[i]) for i in range(len(unique_labels))]

        axs[i].legend(handles, unique_labels, frameon=False, fontsize=4)

    plt.tight_layout()

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\ExM-EM-histograms_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization of ExM and EM histograms plotted and saved')
    plt.close(fig)



    #%% 
    ################################################## PCA PLOTS ##################################################
    ###############################################################################################################
    #Plotting Explained variability across dimensions

    ### For relative counts
    ##Plotting  the square-root eigenvalue spectrum
    fig,axs = plt.subplots(nrows =1, ncols = 1)
    axs.plot(rel_eigvals/np.sum(rel_eigvals) * 100,'-o',color='black')
    axs.set_title(f'{neuron_of_interest}, Eigenvalue spectrum, relative counts syn>={min_desired_count}')
    axs.set_xlabel('Dimensions')
    axs.set_ylabel('Explained-variability (percentage)')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-explained-variability-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA explained variability plotted and saved')
    plt.close(fig)


    ##Plotting PCA1 vs PCA2
    fig, axs = plt.subplots(nrows=1, ncols=1)

    # Labels
    syn_df_grouped = syn_df.groupby(['instance_post', 'dorso-ventral', 'hemisphere']).agg({'W_new': sum})
    syn_df_grouped.reset_index(level=category_column, inplace=True)
    cat_list = syn_df_grouped[category_column].tolist()
    # Scatter plot
    pca_coords = (rel_data_array_norm @ rel_eigvecs[:, 0], rel_data_array_norm @ rel_eigvecs[:, 1])
    # Unique categories (D and V)
    # Unique categories (eg., D and V)
    unique_categories = np.unique(cat_list)
    # Define colors for the categories based on seaborn palette "Set2"
    palette = sns.color_palette(color_cat_set, len(unique_categories))
    # Scatter plot for each category
    for i, category in enumerate(unique_categories):
        category_mask = np.array(cat_list) == category
        x = pca_coords[0][category_mask]
        y = pca_coords[1][category_mask]
        axs.scatter(x, y, color=palette[i], label=category)
    # Set the title, labels, and legend
    axs.set_title(f'{neuron_of_interest}, PCA1 vs PCA2, relative-counts, syn>={min_desired_count}')
    axs.set_xlabel('PCA1')
    axs.set_ylabel('PCA2')
    axs.legend(title=category_column)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA1-PCA2-relative-counts_{dataset_name}_{neuron_of_interest}_{category_column}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA1 vs PCA2 plotted and saved')
    plt.close(fig)



    ## Plotting PCA1, PCA2, PCA3
    fig = plt.figure(figsize=(10, 8))
    axs = fig.add_subplot(111, projection='3d')  # Create a 3D axes object

    # PCA
    # Assuming you have already defined and computed rel_data_array and rel_eigvecs

    # Scatter plot
    pca_coords = (
        rel_data_array_norm @ rel_eigvecs[:, 0],
        rel_data_array_norm @ rel_eigvecs[:, 1],
        rel_data_array_norm @ rel_eigvecs[:, 2]  # Adding the third dimension to PCA coordinates
    )

    # Unique categories (eg., D and V)
    unique_categories = np.unique(cat_list)
    # Define colors for the categories based on seaborn palette "Set2"
    palette = sns.color_palette(color_cat_set, len(unique_categories))
    # Scatter plot for each category
    for i, category in enumerate(unique_categories):
        category_mask = np.array(cat_list) == category
        x = pca_coords[0][category_mask]
        y = pca_coords[1][category_mask]
        z = pca_coords[2][category_mask]
        axs.scatter(x, y, z, color=palette[i], label=category)

    # Set the title, labels, and legend
    axs.set_title(f'{neuron_of_interest}, PCA1 vs PCA2 vs PCA3, relative-counts, syn>={min_desired_count}')
    axs.set_xlabel('PCA1')
    axs.set_ylabel('PCA2')
    axs.set_zlabel('PCA3')
    axs.legend(title=category_column)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA1-PCA2-PCA3-relative-counts_{dataset_name}_{neuron_of_interest}_{category_column}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA1 vs PCA2 vs PCA3 plotted and saved')
    plt.close(fig)

    ##Plotting contribution of neurons to PCA (Eigenvectors)
    threshold = 0.75
    fig, axs = plt.subplots(nrows=1, ncols=1)
    im = axs.imshow(rel_eigvecs, cmap='coolwarm', aspect='auto')

    # Get the shape of the eigenvectors array
    num_neurons, num_pcs = rel_eigvecs.shape

    # Loop over all elements to add text inside the plot if the absolute value is greater than the threshold
    for i in range(num_neurons):
        for j in range(num_pcs):
            value = rel_eigvecs[i, j]
            if abs(value) > threshold:
                # Add text at position (j, i) with the value rounded to 2 decimal places
                axs.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize = 8)

    plt.colorbar(im, ax=axs)
    axs.set_xlabel('Principal components (PCs)')
    axs.set_ylabel('Neurons')
    axs.set_yticks(np.arange(num_neurons))
    axs.set_yticklabels(rel_data.columns)
    plt.title('Contribution of neurons to PCs')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-contribution-from-neurons_relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Contribution of neurons to PCA plotted and saved')
    plt.close(fig)

    ##Plotting THE RELATIVE contribution of neurons to EACH PCA (Normalized eigenvectors)
    ## Normalizing the eigenvectors
    subthreshold = 0.05
    rel_eigvecs_abs = np.absolute(rel_eigvecs)
    norm_eigvecs_abs = rel_eigvecs_abs / rel_eigvecs_abs.sum(axis=0)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    im = axs.imshow(norm_eigvecs_abs, cmap='Greens', aspect='auto')

    # Get the shape of the eigenvectors array
    num_neurons, num_pcs = norm_eigvecs_abs.shape

    # Add text for any value above a max_value-subthreshold:
    for i in range(num_neurons):
        for j in range(num_pcs):
            value = norm_eigvecs_abs[i, j]
            max_value = np.max(norm_eigvecs_abs[:, j])
            if abs(value) > max_value-subthreshold:
                # Add text at position (j, i) with the value rounded to 2 decimal places
                axs.text(j, i, f'{value:.2f}', ha='center', va='center', color=(0.5,0.5,0.5), fontsize = 8)

    # Add text for the maximum value per column
    for j in range(num_pcs):
        max_value = np.max(norm_eigvecs_abs[:, j])
        max_index = np.argmax(norm_eigvecs_abs[:, j])
        axs.text(j, max_index, f'{max_value:.2f}', ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=axs)
    axs.set_xlabel('Principal components (PCs)')
    axs.set_ylabel('Neurons')
    axs.set_yticks(np.arange(num_neurons))
    axs.set_yticklabels(abs_data.columns)
    plt.title('Relative contribution of neurons to PCs')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-relative-contribution-from-neurons_relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Contribution of neurons to PCA plotted and saved')
    plt.close(fig)


    ### For absolute counts
    ##Plotting  the square-root eigenvalue spectrum
    fig,axs = plt.subplots(nrows =1, ncols = 1)
    axs.plot(abs_eigvals/np.sum(abs_eigvals) * 100,'-o',color='black')
    axs.set_title(f'{neuron_of_interest}, Eigenvalue spectrum, absolute counts syn>={min_desired_count}')
    axs.set_xlabel('Dimensions')
    axs.set_ylabel('Explained-variability (percentage)')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-explained-variability-absolute-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA explained variability plotted and saved')
    plt.close(fig)


    ##Plotting PCA1 vs PCA2
    fig, axs = plt.subplots(nrows=1, ncols=1)

    # Labels
    syn_df_grouped = syn_df.groupby(['instance_post', 'dorso-ventral', 'hemisphere']).agg({'W_new': sum})
    syn_df_grouped.reset_index(level=category_column, inplace=True)
    cat_list = syn_df_grouped[category_column].tolist()
    # Scatter plot
    pca_coords = (abs_data_array_norm @ abs_eigvecs[:, 0], abs_data_array_norm @ abs_eigvecs[:, 1])
    # Unique categories (D and V)
    # Unique categories (eg., D and V)
    unique_categories = np.unique(cat_list)
    # Define colors for the categories based on seaborn palette "Set2"
    palette = sns.color_palette(color_cat_set, len(unique_categories))
    # Scatter plot for each category
    for i, category in enumerate(unique_categories):
        category_mask = np.array(cat_list) == category
        x = pca_coords[0][category_mask]
        y = pca_coords[1][category_mask]
        axs.scatter(x, y, color=palette[i], label=category)
    # Set the title, labels, and legend
    axs.set_title(f'{neuron_of_interest}, PCA1 vs PCA2, absolute-counts, syn>={min_desired_count}')
    axs.set_xlabel('PCA1')
    axs.set_ylabel('PCA2')
    axs.legend(title=category_column)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA1-PCA2-absolute-counts_{dataset_name}_{neuron_of_interest}_{category_column}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA1 vs PCA2 plotted and saved')
    plt.close(fig)



    ## Plotting PCA1, PCA2, PCA3
    fig = plt.figure(figsize=(10, 8))
    axs = fig.add_subplot(111, projection='3d')  # Create a 3D axes object

    # PCA
    # Assuming you have already defined and computed rel_data_array and rel_eigvecs

    # Scatter plot
    pca_coords = (
        abs_data_array_norm @ abs_eigvecs[:, 0],
        abs_data_array_norm @ abs_eigvecs[:, 1],
        abs_data_array_norm @ abs_eigvecs[:, 2]  # Adding the third dimension to PCA coordinates
    )

    # Unique categories (eg., D and V)
    unique_categories = np.unique(cat_list)
    # Define colors for the categories based on seaborn palette "Set2"
    palette = sns.color_palette(color_cat_set, len(unique_categories))
    # Scatter plot for each category
    for i, category in enumerate(unique_categories):
        category_mask = np.array(cat_list) == category
        x = pca_coords[0][category_mask]
        y = pca_coords[1][category_mask]
        z = pca_coords[2][category_mask]
        axs.scatter(x, y, z, color=palette[i], label=category)

    # Set the title, labels, and legend
    axs.set_title(f'{neuron_of_interest}, PCA1 vs PCA2 vs PCA3, absolute-counts, syn>={min_desired_count}')
    axs.set_xlabel('PCA1')
    axs.set_ylabel('PCA2')
    axs.set_zlabel('PCA3')
    axs.legend(title=category_column)

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA1-PCA2-PCA3-absolute-counts_{dataset_name}_{neuron_of_interest}_{category_column}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: PCA1 vs PCA2 plotted and saved')
    plt.close(fig)

    ##Plotting contribution of neurons to PCA (Eigenvectors)
    threshold = 0.75
    fig, axs = plt.subplots(nrows=1, ncols=1)
    im = axs.imshow(abs_eigvecs, cmap='coolwarm', aspect='auto')

    # Get the shape of the eigenvectors array
    num_neurons, num_pcs = abs_eigvecs.shape

    # Loop over all elements to add text inside the plot if the absolute value is greater than the threshold
    for i in range(num_neurons):
        for j in range(num_pcs):
            value = abs_eigvecs[i, j]
            if abs(value) > threshold:
                # Add text at position (j, i) with the value rounded to 2 decimal places
                axs.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize = 8)

    plt.colorbar(im, ax=axs)
    axs.set_xlabel('Principal components (PCs)')
    axs.set_ylabel('Neurons')
    axs.set_yticks(np.arange(num_neurons))
    axs.set_yticklabels(abs_data.columns)
    plt.title('Contribution of neurons to PCs')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-contribution-from-neurons_absolte-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Contribution of neurons to PCA plotted and saved')
    plt.close(fig)

    ##Plotting THE RELATIVE contribution of neurons to EACH PCA (Normalized eigenvectors)
    ## Normalizing the eigenvectors
    subthreshold = 0.05
    abs_eigvecs_abs = np.absolute(abs_eigvecs)
    norm_eigvecs_abs = abs_eigvecs_abs / abs_eigvecs_abs.sum(axis=0)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    im = axs.imshow(norm_eigvecs_abs, cmap='Greens', aspect='auto')

    # Get the shape of the eigenvectors array
    num_neurons, num_pcs = norm_eigvecs_abs.shape

    # Add text for any value above a max_value-subthreshold:
    for i in range(num_neurons):
        for j in range(num_pcs):
            value = norm_eigvecs_abs[i, j]
            max_value = np.max(norm_eigvecs_abs[:, j])
            if abs(value) > max_value-subthreshold:
                # Add text at position (j, i) with the value rounded to 2 decimal places
                axs.text(j, i, f'{value:.2f}', ha='center', va='center', color=(0.5,0.5,0.5), fontsize = 8)

    # Add text for the maximum value per column
    for j in range(num_pcs):
        max_value = np.max(norm_eigvecs_abs[:, j])
        max_index = np.argmax(norm_eigvecs_abs[:, j])
        axs.text(j, max_index, f'{max_value:.2f}', ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=axs)
    axs.set_xlabel('Principal components (PCs)')
    axs.set_ylabel('Neurons')
    axs.set_yticks(np.arange(num_neurons))
    axs.set_yticklabels(abs_data.columns)
    plt.title('Relative contribution of neurons to PCs')

    #Plot saving
    if save_figures:
        save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
        figure_title = f'\PCA-relative-contribution-from-neurons_absolte-counts_{dataset_name}_{neuron_of_interest}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Contribution of neurons to PCA plotted and saved')
    plt.close(fig)




    #%% 
    ############################################### NEUROPIL - PLOTS ############################################
    #############################################################################################################

    # Tm9: '#458A7C', Tm1:'#D57DA3'

    ################################################ BINARY COUNTs ###############################################
    #Plotting a single neuron

    _data = binary_df[presence_threshold_sorted_column_order] # FIlter abnd sorting

    #Gettting the center point in specific neuropile from database
    xyz_neuropil = 'XYZ-ME'
    xyz_df = database_df[database_df['Updated_seg_id'].isin(root_ids)].copy()
    xyz_pre = xyz_df[xyz_neuropil].tolist()
    # Split each string by comma and convert the elements to floats
    xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
    xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])

    #set(root_ids).difference(database_df['Updated_seg_id'])



    # #Seb commented out since we plot all neurons in the next step anyways
    # #Getting list for dot sizes and colors based on instance counts of a pre_partner
    # pre_partner = 'OA-AL2b2-R1'
    # #Dot sizes
    # dot_sizes = _data[pre_partner].fillna(0).tolist()
    # dot_sizes_ME = [size*5 for size in dot_sizes]  # Increase size by a factor of 20
    # dot_sizes_LO = [size*5 for size in dot_sizes]  # Increase size by a factor of 10
    # OL_R = flywire.get_neuropil_volumes([mesh_OL_L]) #['ME_R','LO_R','LOP_R']
    # fig = plt.figure(figsize=20,10)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME,c=neuron_color)  # Adjust the size (s) as desired
    # ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=1,c='k',alpha=1) # All dots
    # navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')
    # ax.azim = mesh_azim_L 
    # ax.elev = mesh_elev_L
    # ax.set_title(f"Presynaptic partner: {pre_partner}, {mesh_OL_L}", fontsize = 8)

    # #Plot saving
    # if save_figures:
    #     save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
    #     figure_title = f'\Meshes_XYZ_positions_ME_binary_{dataset_name}_{mesh_OL_L}_{pre_partner}_{neuron_of_interest}.pdf'
    #     fig.savefig(save_path+figure_title)
    #     print('FIGURE: Visualization of XYZ positions plotted and saved')
    # plt.close(fig)



    ################################################ BINARY COUNTs ###############################################
    # Plotting all neurons in the same pdf page
    # Data
    _data = binary_df[presence_threshold_sorted_column_order] # FIlter abnd sorting

    # Assuming pre_partner_list is a list of objects to be plotted
    pre_partner_list = _data.columns.tolist()
    OL_R = flywire.get_neuropil_volumes([mesh_ME])

    # Create a PDF file to save the plots
    save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Meshes_XYZ_positions_ME_binary_all_partners_{dataset_name}_{neuron_of_interest}.pdf'
    outputPath =  save_path + figure_title
    pdf_pages = PdfPages(outputPath)

    # Calculate the number of rows and columns for the grid layout
    num_plots = len(pre_partner_list)
    num_cols = 4  # Adjust the number of columns as needed
    num_rows = (num_plots - 1) // num_cols + 1

    # Set the figure size based on DIN4 page size
    fig_width = 8.27 *2  # Width of DIN4 page in inches
    fig_height = 11.69  *2 # Height of DIN4 page in inches

    # Calculate the size of each subplot
    subplot_width = fig_width / num_cols * 4  # Adjust the multiplier as needed
    subplot_height = fig_height / num_rows * 4  # Adjust the multiplier as needed

    # Calculate the size of the plotted content
    content_width = subplot_width * 0.9  # Adjust the multiplier as needed
    content_height = subplot_height * 0.9  # Adjust the multiplier as needed

    # Create the figure and subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), subplot_kw={'projection': '3d'})

    # Set the size of the plotted content in each subplot
    for ax in axes.flatten():
        ax.set_box_aspect([content_width, content_height, content_height])

    # Flatten the axes array if it's a 1D array
    if num_plots == 1:
        axes = [axes]

    # Loop through the objects and create subplots
    for i, (pre_partner, ax) in enumerate(zip(pre_partner_list, axes.flatten())):
        # Generate the plot for the current object
        dot_sizes = _data[pre_partner].fillna(0).tolist()
        dot_sizes_ME = [size * 10 for size in dot_sizes]  # Increase size by a factor of X

        # Plot the object
        ax.scatter(
            xyz_pre_arr_new[:, 0],
            xyz_pre_arr_new[:, 1],
            xyz_pre_arr_new[:, 2],
            s=dot_sizes_ME,
            c=neuron_color,
            alpha=0.9
        )
        ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=2,c='k',alpha=1) # All dots
        navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um') #

        # Rotating the view
        ax.azim = mesh_azim 
        ax.elev = mesh_elev 

        # Set plot title
        ax.set_title(f"Presynaptic partner: {pre_partner}", fontsize = 8)

        # Remove ticks and tick labels from XYZ axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Remove the spines (axis lines)
        ax.spines['left'].set_visible(False)

        # Remove axes lines
        ax.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_xaxis.line.set_linewidth(0.0)
        ax.w_yaxis.line.set_linewidth(0.0)
        ax.w_zaxis.line.set_linewidth(0.0)

        # Remove background
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Hide axis lines and tick markers
        ax.w_xaxis.line.set_color("none")
        ax.w_yaxis.line.set_color("none")
        ax.w_zaxis.line.set_color("none")
        ax.set_axis_off()

        # Remove grid lines
        ax.grid(False)

        # Remove the axis marker
        ax._axis3don = False

        ax.axis('off')

        # Add any additional customization to the plot

    # Remove empty subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    # Adjust the spacing between subplots and between the title and the plot
    fig.subplots_adjust(wspace=0, hspace=0, top=0.85)

    # Save the figure with subplots to the PDF file
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0)

    # Close the figure and PDF file
    plt.close(fig)
    pdf_pages.close()

    print(f"Plots saved in {outputPath}")




    ################################################ INSTANCE COUNTS ##############################################
    ## Visualizing instance (copies of the same neuron type) counts
    # Plots for single neurons of interest

    #Gettting the center point in specific neuropile from database
    xyz_neuropil = 'XYZ-ME'
    xyz_df = database_df[database_df['Updated_seg_id'].isin(root_ids)].copy()
    xyz_pre = xyz_df[xyz_neuropil].tolist()
    # Split each string by comma and convert the elements to floats
    xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
    xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])


    # Plotting all neurons in the same pdf page
    # Data
    _data = counting_instances_df.T[presence_threshold_sorted_column_order] # FIlter abnd sorting

    # Assuming pre_partner_list is a list of objects to be plotted
    pre_partner_list = _data.columns.tolist()
    OL_R = flywire.get_neuropil_volumes([mesh_ME])

    # Create a PDF file to save the plots
    save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' # Your path
    figure_title = f'\Meshes_XYZ_positions_ME_instance_counts_all_partners_{dataset_name}_{neuron_of_interest}.pdf'
    outputPath =  save_path + figure_title
    pdf_pages = PdfPages(outputPath)

    # Calculate the number of rows and columns for the grid layout
    num_plots = len(pre_partner_list)
    num_cols = 4  # Adjust the number of columns as needed
    num_rows = (num_plots - 1) // num_cols + 1

    # Set the figure size based on DIN4 page size
    fig_width = 8.27 *2  # Width of DIN4 page in inches
    fig_height = 11.69  *2 # Height of DIN4 page in inches

    # Calculate the size of each subplot
    subplot_width = fig_width / num_cols * 4  # Adjust the multiplier as needed
    subplot_height = fig_height / num_rows * 4  # Adjust the multiplier as needed

    # Calculate the size of the plotted content
    content_width = subplot_width * 0.9  # Adjust the multiplier as needed
    content_height = subplot_height * 0.9  # Adjust the multiplier as needed

    # Create the figure and subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), subplot_kw={'projection': '3d'})

    # Set the size of the plotted content in each subplot
    for ax in axes.flatten():
        ax.set_box_aspect([content_width, content_height, content_height])

    # Flatten the axes array if it's a 1D array
    if num_plots == 1:
        axes = [axes]

    # Loop through the objects and create subplots
    for i, (pre_partner, ax) in enumerate(zip(pre_partner_list, axes.flatten())):
        # Generate the plot for the current object
        dot_sizes = _data[pre_partner].fillna(0).tolist()
        dot_sizes_ME = [size * 5 for size in dot_sizes]  # Increase size by a factor of X

        size_color_map = {}
        color_palette = sns.color_palette(color_palette_name, int(max(counting_instances_df.max()))) # before:  sns.color_palette(color_palette_name, len(set(dot_sizes))) 

        #Dot colors
        dot_colors = []

        
        for size in dot_sizes:
            if size != 0.0 and size not in size_color_map:
                size_color_map[size] = color_palette[len(size_color_map)] #TODO Think how to improve this line to fix the colors!
            #dot_colors.append(size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0))
            color = size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0)
            color = (*color[:3], 1.0)  # Make color fully opaque
            dot_colors.append(color)

        # Plot the object
        ax.scatter(
            xyz_pre_arr_new[:, 0],
            xyz_pre_arr_new[:, 1],
            xyz_pre_arr_new[:, 2],
            s=dot_sizes_ME,
            c=dot_colors,
            alpha=0.9
        )
        ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=1,c='k',alpha=1) # All dots
        navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um') #

        # Rotating the view
        ax.azim = mesh_azim 
        ax.elev = mesh_elev 

        # Set plot title
        ax.set_title(f"Presynaptic partner: {pre_partner}", fontsize = 8)

        # Remove ticks and tick labels from XYZ axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Remove the spines (axis lines)
        ax.spines['left'].set_visible(False)

        # Remove axes lines
        ax.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.w_xaxis.line.set_linewidth(0.0)
        ax.w_yaxis.line.set_linewidth(0.0)
        ax.w_zaxis.line.set_linewidth(0.0)

        # Remove background
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Hide axis lines and tick markers
        ax.w_xaxis.line.set_color("none")
        ax.w_yaxis.line.set_color("none")
        ax.w_zaxis.line.set_color("none")
        ax.set_axis_off()

        # Remove grid lines
        ax.grid(False)

        # Remove the axis marker
        ax._axis3don = False

        ax.axis('off')

        # Add any additional customization to the plot

    # Remove empty subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    # Adjust the spacing between subplots and between the title and the plot
    fig.subplots_adjust(wspace=0, hspace=0, top=0.85)

    # Save the figure with subplots to the PDF file
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0)

    # Close the figure and PDF file
    plt.close(fig)
    pdf_pages.close()

    print(f"Plots saved in {outputPath}")



##############################################################################################################
#####################################    OLD (perhaps useful) CODE BELOW    ##################################
##############################################################################################################

#Temp CODE, Debugging

# tem_df = pd.read_csv(r'E:\Connectomics-Data\FlyWire\Excels\optic_lobe_ids_left.csv')
# ol_id_list = tem_df.columns.tolist()

# def non_match_elements(list_a, list_b):
#      non_match = []
#      for i in list_a:
#          if i not in list_b:
#              non_match.append(i)
#      return non_match


# non_match = non_match_elements(ol_id_list, id_column,)
# print("No match elements: ", non_match)

# %%
# ################################################ BINARY COUNTs ###############################################
# ################################# PLOTTING EACH  NEURON IN DIFFERENT PDF PAGES ###############################


# from matplotlib.backends.backend_pdf import PdfPages
# # Assuming pre_partner_list is a list of objects to be plotted
# Data
# _data = binary_df[presence_threshold_sorted_column_order] # FIlter abnd sorting

# pre_partner_list = _data.columns.tolist()
# OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']


# # Create a PDF file to save the plots
# save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
# figure_title = f'\Meshes_XYZ_positions_ME_binary_{dataset_name}_{neuron_of_interest}_many_pages.pdf'
# outputPath =  save_path+figure_title
# pdf_pages = PdfPages(outputPath)

# # Loop through the objects and create subplots
# for i, pre_partner in enumerate(pre_partner_list):
#     # Generate the plot for the current object
#     dot_sizes = _data[pre_partner].fillna(0).tolist()
#     dot_sizes_ME = [size*20 for size in dot_sizes]  # Increase size by a factor of 20
#     dot_sizes_LO = [size*10 for size in dot_sizes]  # Increase size by a factor of 10

#     fig = plt.figure(figsize=(8.27, 11.69))  # DIN4 page size (in inches)
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the object
#     ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME, c=neuron_color, alpha=0.9)
#     navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')

#     #Rotating the view
#     ax.azim = mesh_azim 
#     ax.elev = mesh_elev 

#     # Set plot title
#     ax.set_title(f"Presynaptic partner: {pre_partner}")

#     # Add any additional customization to the plot

#     # Save the current plot as a subplot in the PDF file
#     pdf_pages.savefig(fig, bbox_inches='tight')

#     # Close the current figure
#     plt.close(fig)

# # Save and close the PDF file
# pdf_pages.close()

# print(f"Plots saved in {outputPath}")


# %%
# ########################################### DIMENTIONALITY Reduction ##########################################
# ###############################################################################################################
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# ### PCA across columns - relative counts

# # Define number of clusters (not mandatory for PCA, just for exploration)
# num_clusters = 2

# #Defining range of neurons (featured) to plot based on rank (defining the subset for the PCA)
# features = presence_threshold_sorted_column_order # filtering based on " presence_threshold"

# #Get the features data
# curr_rel_df = syn_popularity_rel_df[presence_threshold_sorted_column_order].copy()
# curr_rel_df_no_NaNs = curr_rel_df.fillna(0).copy() #replacing NaNs with zero
# rel_data = curr_rel_df_no_NaNs[features].copy()


# #Run PCA on the data and reduce the dimensions in pca_num_components dimensions
# #Across columns
# pca_num_components = 2
# reduced_rel_data = PCA(n_components=pca_num_components).fit_transform(rel_data)
# results_rel_columns = pd.DataFrame(reduced_rel_data,columns=['pca1','pca2'])


# #Adding cluster id column
# clustering_kmeans = KMeans(n_clusters=num_clusters)
# rel_data['clusters'] = clustering_kmeans.fit_predict(rel_data)
# results_rel_columns['clusters'] = rel_data['clusters'].tolist()

# #Adding column id column
# rel_data['column_id'] = [x[4:5] for x in rel_data.index.tolist()]
# results_rel_columns['column_id'] = [x[4:5] for x in rel_data.index.tolist()]

# ### PCA across columns - absolute counts

# # Define number of clusters (not mandatory for PCA, just for exploration)
# num_clusters = 2

# #Defining range of neurons (featured) to plot based on rank (defining the subset for the PCA)
# features = presence_threshold_sorted_column_order # filtering based on " presence_threshold"

# #Get the features data
# curr_abs_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy()
# curr_abs_df_no_NaNs = curr_abs_df.fillna(0).copy() #replacing NaNs with zero
# abs_data = curr_abs_df_no_NaNs[features].copy()


# #Run PCA on the data and reduce the dimensions in pca_num_components dimensions
# #Across columns
# pca_num_components = 2
# reduced_abs_data = PCA(n_components=pca_num_components).fit_transform(abs_data)
# results_abs_columns = pd.DataFrame(reduced_abs_data,columns=['pca1','pca2'])


# #Adding cluster id column
# clustering_kmeans = KMeans(n_clusters=num_clusters)
# abs_data['clusters'] = clustering_kmeans.fit_predict(abs_data)
# results_abs_columns['clusters'] = abs_data['clusters'].tolist()

# #Adding column id column
# abs_data['column_id'] = [x[4:5] for x in abs_data.index.tolist()]
# results_abs_columns['column_id'] = [x[4:5] for x in abs_data.index.tolist()]



# %%
# ################################################## PCA PLOTS ##################################################
# #Plotting PCA plots

# #Figure
# fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(30*cm, 30*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
# fig.tight_layout(pad=8) # Adding some space between subplots

# #First Axis
# sns.scatterplot(x="pca1", y="pca2", data=results_rel_columns, ax = axs[0])
# axs[0].set_title(f'{neuron_of_interest}, PCA - count % of popular neurons (syn>={min_desired_count})')
# axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# #Next Axis
# sns.scatterplot(x="pca1", y="pca2", data=results_abs_columns, ax = axs[1])
# axs[1].set_title(f'{neuron_of_interest}, PCA - absolute count of popular neurons (syn>={min_desired_count})')
# axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)



# #Plot saving
# if save_figures:
#     save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
#     figure_title = f'\PCA-relative-subset-connectivity_{dataset_name}_{neuron_of_interest}.pdf'
#     fig.savefig(save_path+figure_title)
#     print('FIGURE: PCA plotted and saved')
# plt.close(fig)


# %%
# ######################################### DISTRIBUTIONS FROM PERMUTATIONS ########################################

# # Valid code:
# def permutation_test_old(cluster_df, column1_name, column2_name, num_permutations):
#     observed_corr = cluster_df[column1_name].corr(cluster_df[column2_name])  # Compute the observed correlation
#     shuffled_corrs = []

#     for _ in range(num_permutations):
#         shuffled_values = cluster_df[column2_name].sample(frac=1).values  # Shuffle the values of the second column
#         shuffled_df = pd.DataFrame({column1_name: cluster_df[column1_name].values,
#                                     f"Shuffled_{column2_name}": shuffled_values})
#         shuffled_corr = shuffled_df[column1_name].corr(shuffled_df[f"Shuffled_{column2_name}"])
#         shuffled_corrs.append(shuffled_corr)

#     # Calculate the p-value based on the number of shuffled correlations larger or equal to the observed correlation
#     p_value = (np.sum(np.abs(shuffled_corrs) >= np.abs(observed_corr)) + 1) / (num_permutations + 1)

#     return observed_corr, p_value, shuffled_corrs

# # Data
# ### Absolute counts
# curr_df = syn_popularity_abs_df[presence_threshold_sorted_column_order].copy() #  filtering based on " presence_threshold"
# curr_df = curr_df.fillna(0).copy()
# # Columns to be compared
# column_names = curr_df.columns.tolist()

# # In case you want to run it for different subsets of the data, code need to be modify for defining column_names
# clusters_list = [curr_df]  # Insert your actual cluster DataFrames

# # Create subplots to visualize the results
# valid_pairs = list(combinations(column_names, 2))

# # Define the number of subplots per figure
# subplots_per_figure = 15
# num_figures = int(np.ceil(len(valid_pairs) / subplots_per_figure))

# # Create and display figures with subplots
# for fig_num in range(num_figures):
#     start_idx = fig_num * subplots_per_figure
#     end_idx = min((fig_num + 1) * subplots_per_figure, len(valid_pairs))

#     num_rows = num_cols = int(np.ceil(np.sqrt(end_idx - start_idx)))
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

#     for pair_idx, (column1_name, column2_name) in enumerate(valid_pairs[start_idx:end_idx]):
#         observed_corr, p_value, shuffled_corrs = permutation_test(curr_df, column1_name, column2_name, num_permutations)

#         # Plot the observed correlation and the distribution of shuffled correlations in the corresponding subplot
#         ax = axes[pair_idx // num_cols, pair_idx % num_cols]
#         ax.hist(shuffled_corrs, bins=30, alpha=0.6, color='gray', edgecolor='black', label='Shuffled')
#         ax.axvline(observed_corr, color='red', linestyle='dashed', linewidth=2, label='Observed')
#         ax.set_title(f"{column1_name} vs. {column2_name}")
#         ax.legend()

#         # Annotate the p-value in the plot
#         ax.text(0.8, 0.85, f"P-value: {p_value:.3f}", transform=ax.transAxes, fontsize=10, fontweight='bold')

#     # Hide empty subplots if any
#     for pair_idx in range(end_idx - start_idx, num_rows * num_cols):
#         axes[pair_idx // num_cols, pair_idx % num_cols].axis('off')

#     plt.tight_layout()
#     plt.show()


# # #Option 2: ( for correlation matrices)
# # Initialize an empty DataFrame to store the correlation matrix
# correlation_abs_no_NaN_df = pd.DataFrame(columns=curr_df.columns, index=curr_df.columns)
# p_values_correlation_abs_no_NaN_df = pd.DataFrame(columns=curr_df.columns, index=curr_df.columns)

# # Calculate the correlation matrix using Pearson correlation
# for col1, col2 in combinations(curr_df.columns, 2):
#     # Get the data for the current pair of columns
#     x_data, y_data = curr_df[col1], curr_df[col2]
    
#     # Compute the Pearson correlation coefficient and p-value
#     correlation_coefficient, p_value = pearsonr(x_data, y_data)
    
#     # Store the absolute value of the correlation coefficient in the DataFrame
#     correlation_abs_no_NaN_df.at[col1, col2] = correlation_coefficient
#     correlation_abs_no_NaN_df.at[col2, col1] = correlation_coefficient
#     p_values_correlation_abs_no_NaN_df.at[col1, col2] = p_value
#     p_values_correlation_abs_no_NaN_df.at[col2, col1] = p_value

# # Fill the diagonal with 1.0 since the correlation of a feature with itself is always 1
# np.fill_diagonal(correlation_abs_no_NaN_df.values, 1.0)


# if instance_count_plot:
#     #Getting list for dot sizes and colors based on instance counts of a pre_partner
#     pre_partner = 'Dm12'

#     #Dot sizes
#     dot_sizes = counting_instances_df.T[pre_partner].fillna(0).tolist()
#     dot_sizes_ME = [size*20 for size in dot_sizes]  # Increase size by a factor of 20
#     dot_sizes_LO = [size*10 for size in dot_sizes]  # Increase size by a factor of 10

#     size_color_map = {}
#     color_palette = sns.color_palette(color_palette_name, int(max(counting_instances_df.max()))) # before:  sns.color_palette(color_palette_name, len(set(dot_sizes))) 

#     #Dot colors
#     dot_colors = []

    
#     for size in dot_sizes:
#         if size != 0.0 and size not in size_color_map:
#             size_color_map[size] = color_palette[len(size_color_map)] #TODO Think how to improve this line to fix the colors!
#         #dot_colors.append(size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0))
#         color = size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0)
#         color = (*color[:3], 1.0)  # Make color fully opaque
#         dot_colors.append(color)




#     OL_R = flywire.get_neuropil_volumes([mesh_ME]) #['ME_R','LO_R','LOP_R']
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME,c=dot_colors)  # Adjust the size (s) as desired
#     ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=5,c='k') # All dots
#     navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um') #
#     ax.azim = mesh_azim 
#     ax.elev = mesh_elev 
#     #plt.show()

#     #Plot saving
#     if save_figures:
#         save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
#         figure_title = f'\Meshes_XYZ_positions_instance_counts_ME_{dataset_name}_{pre_partner}_{neuron_of_interest}.pdf'
#         fig.savefig(save_path+figure_title)
#         print('FIGURE: Visualization of XYZ positions plotted and saved')
#     plt.close(fig)

#     #Gettting the center point in specific neuropile from database
#     xyz_neuropil = 'XYZ-LO'
#     xyz_df = database_df[database_df['seg_id'].isin(root_ids)].copy()
#     xyz_pre = xyz_df[xyz_neuropil].tolist()
#     # Split each string by comma and convert the elements to floats
#     xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
#     xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])

#     OL_R = flywire.get_neuropil_volumes([mesh_LO]) #['ME_R','LO_R','LOP_R']

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_LO,c=dot_colors)  # Adjust the size (s) as desired
#     navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')
#     ax.azim = -6
#     ax.elev = -57
#     #plt.show()

#     #Plot saving
#     if save_figures:
#         save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots' #r'D:\Connectomics-Data\FlyWire\Pdf-plots' 
#         figure_title = f'\Meshes_XYZ_positions_instance_counts_LO_{dataset_name}_{pre_partner}_{neuron_of_interest}.pdf'
#         fig.savefig(save_path+figure_title)
#         print('FIGURE: Visualization of XYZ positions plotted and saved')
#     plt.close(fig)

