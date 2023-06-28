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
from scipy.stats import pearsonr
import os
import math 
from caveclient import CAVEclient
client = CAVEclient('flywire_fafb_production')

#%% Plots settings
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

cm = 1/2.54  # centimeters in inches
save_figures = False



#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################
#General analysis information (user defined)


#Count coverage
desired_coverage = 80 # in percent
plus_minus = 5 # in percent

#Synaptic counts
desired_count = 3 # minimun number of synapses to consider in the analysis
syn_thr_min = 1 # minimun number of synaptic contacts to be considered as relevant for the analysis
num_type_copies_min = 2 #Number for neuron copies relevant for the low-synapses partners
presence_threshold = 0.05 # If a partner is in less than 5% of the columns, it will be discarted for further visualizations

#Neuron counts
desired_quantile = 0.8 # selected quantile for neuron counts

#Main presynaptic partner
last_input_neuron = 1000 # last input to be considered in summary dataframes across columns

#Data set 
dataset_name = 'FAFB'
neuron_of_interest = 'Tm9' 
instance_id_column = 'optic_lobe_id' # 'optic_lobe_id', 'column_id'

#Path and file
dataPath =  r'E:\Connectomics-Data\FlyWire\Excels\drive-data-sets'
fileDate = '20230621'
fileName = f'Tm9_neurons_input_count_ME_L_{fileDate}.xlsx'
fileName_database = f'Tm9 proofreadings_{fileDate}.xlsx'

#Expansion microscopy
ExM_dataPath =  r'E:\Connectomics-Data\FlyWire\Excels\expansion-microscopy'


#%% 
################################################## PRE-ANALYSIS ###############################################
############################################# ELECTRON MICROSCOPY (EM) ########################################

#Loading FAFB data from our data base (excel file)
filePath = os.path.join(dataPath,fileName)
df = pd.read_excel(filePath)

#Loading FAFB data from our data base (excel file)
filePath = os.path.join(dataPath,fileName_database)
database_df = pd.read_excel(filePath)

#Dropping rows:
if df["postsynaptic_ID"][0] == 'asdf': #Dropping the fisrt row ('asdf' was added as a walk-around to set that column values as type str)
    df = df.iloc[1: , :]
    df.reset_index(inplace=True,drop=True)

if database_df["seg_id"][0] == 'asdf': #Dropping the fisrt row ('asdf' was added as a walk-around to set that column values as type str)
    database_df = database_df.iloc[1: , :]
    database_df.reset_index(inplace=True,drop=True)

if 'INPUTS PROOFREAD' in df.values: # Removing unnecessary rows ment for human use only
    df = df[df['presynaptic_ID']!= 'INPUTS PROOFREAD'].copy() # Getting rid of info rows with no connectomics data

if 'N.I.' in df.values: # Removing non identified (N.I.) inputs
    df = df[df['symbol']!= 'N.I.'].copy() # Getting rid of info rows with no connectomics data

#Adjusting column names to meet the naming of the FIB25 data sets (interesting for future use or comparison)
#Creating new columns
df['instance_pre'] = df['symbol'] + '::' + df[instance_id_column]
df['instance_post'] = neuron_of_interest + '::' + df[instance_id_column]
df['type_post'] = neuron_of_interest
df['counts']= df['counts'].astype(int)
#Sorting rows based on count synapse number
df = df.groupby(['instance_post']).apply(lambda x: x.sort_values(['counts'], ascending = False)).reset_index(drop=True)
#Ranking the neurons
df['rank'] = df.groupby(['instance_post']).cumcount().tolist()
#Renaming columns
df.rename(columns={'presynaptic_ID':'bodyId_pre', 'counts':'W', 'postsynaptic_ID':'bodyId_post','symbol':'type_pre'}, inplace = True)
#Keeping only columns of interest
cols_to_keep = ['rank','patch_id','column_id','optic_lobe_id','detached_lamina (Y/N)','healthy_L3 (Y/N)','instance_pre','type_pre','bodyId_pre','instance_post','type_post','bodyId_post','W']
df = df[cols_to_keep].copy()
#Filtering out faulty data
df = df[df['detached_lamina (Y/N)'] == 'N'].copy() #Keep only the columns below a healthy lamina
df = df[df['healthy_L3 (Y/N)'] != 'N'].copy() #Discard the onces with clear unhealthy L3

#Calculating relative counts (weigths)
df['column_percent'] = round((df['W'] / df.groupby('instance_post')['W'].transform('sum'))*100,2)
df['cumulative_column_percent'] = df.groupby('instance_post')['column_percent'].cumsum()

#Printing useful information
id_column = df[instance_id_column].unique().tolist()
print(f'The following column ids (n={len(id_column)}) are part of the analysis: \n {id_column}')


desired_count_df = df[df['W']== desired_count].copy()
last_percent_with_desired_count = pd.DataFrame(desired_count_df.groupby(['instance_post'])['cumulative_column_percent'].max()) #Covarage across columns per the desired_count
print(last_percent_with_desired_count)
print(f'Coverage (%) for syn >= {desired_count}')
print(f"The desired {desired_count} have a desired % coverage average across columns = {round(last_percent_with_desired_count['cumulative_column_percent'].mean(),2)}: ")

#Getting the neuron´s meshes
root_ids = df['bodyId_post'].unique().tolist()
m_all = flywire.get_mesh_neuron(root_ids)
print('- Got all neuron meshes -')

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
    curr_df.rename(columns={'counts':'W','symbol':'type_pre'}, inplace = True)
    curr_df['instance_post'] = neuron_of_interest + '::' + curr_df['column_id']
    curr_df['fly_ID'] = curr_df['fly_ID'] + '-ExM'
    # Convert "W" column to integer
    curr_df["W"] = curr_df["W"].astype(int)
    curr_df = curr_df[curr_df['W'] >= desired_count].copy()
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
    N_sum = curr_df['W'].sum()
    N_percentatge = curr_df['W'].tolist()/N_sum * 100
    curr_df['W_percentatge'] = N_percentatge.round(2) #rounding to the second decimal place

    #Synaptic strengh filter
    curr_df = curr_df[curr_df['W']>=syn_thr_min].copy()
    curr_df = curr_df[curr_df['W']>=desired_count].copy()
    #print(f"Input coverage with threshold {syn_thr_min}: {round(curr_df['W_percentatge'].sum(),2)} %")

    #For table across columns
    identity_dict[curr_df['instance_post'][0]] = curr_df['instance_pre'][0:last_input_neuron]
    identity_df= pd.DataFrame(identity_dict) # Here it concatenates at every loop
    identity_type_dict[curr_df['instance_post'][0]] = curr_df['type_pre'][0:last_input_neuron]
    identity_type_df= pd.DataFrame(identity_type_dict) # Here it concatenates at every loop
    identity_type_middle_rank_dict[curr_df['instance_post'][0]] = curr_df['type_pre'][last_input_neuron:]
    identity_type_middle_rank_df= pd.DataFrame(identity_type_middle_rank_dict) # Here it concatenates at every loop
    
    
    #print(f"Input coverage up to the {_end}th input: {round(curr_df['W_percentatge'][0:7].sum(),2)} %")
    abs_connections_dict[curr_df['instance_post'][0]] = curr_df['W'][0:last_input_neuron]
    rel_connections_dict[curr_df['instance_post'][0]] = curr_df['W_percentatge'][0:last_input_neuron]
    abs_connections_df= pd.DataFrame(abs_connections_dict) # Here it concatenates at every loop
    rel_connections_df= pd.DataFrame(rel_connections_dict) # Here it concatenates at every loop
    
    #For scatter plots
    index_name_ls = index_name_ls + ([instance] * len(curr_df['W'][0:last_input_neuron]))
    input_rank_ls = input_rank_ls + list(range(0,len(curr_df['W'][0:last_input_neuron]))) #Concatenating lists across loops
    abs_connection_ls = abs_connection_ls + curr_df['W'][0:last_input_neuron].tolist() #Concatenating lists across loops
    rel_connection_ls = rel_connection_ls + curr_df['W_percentatge'][0:last_input_neuron].tolist() #Concatenating lists across loops
    
#Adding total sums inforamtion to some dataframes
rel_connections_df.loc['Total',:] = rel_connections_df.sum(axis=0).tolist()
rel_connections_df
abs_connections_df.loc['Total',:] = abs_connections_df.sum(axis=0).tolist()
abs_connections_df
    

################################################# RANKS #######################################################
###############################################################################################################
#Analysis across ranks
rank_df = pd.DataFrame(index=index_name_ls)
rank_df ['Abs_connection'] = abs_connection_ls
rank_df ['Rel_connection'] = rel_connection_ls
rank_df ['Connection_rank'] = input_rank_ls

mean_abs_ls = [] # For variability analysis 
mean_rel_ls = [] # For variability analysis 
std_abs_ls = [] # For variability analysis 
std_rel_ls = [] # For variability analysis 
CV_abs_ls = [] # For variability analysis (CV = coefficient of variation)
CV_rel_ls = [] # For variability analysis (CV = coefficient of variation)
p50_abs_ls = [] # For variability analysis 
p50_rel_ls = [] # For variability analysis 

for rank in rank_df['Connection_rank'].unique():
    
    curr_df = rank_df[rank_df['Connection_rank'] == rank].copy()
    #Variability indexes
    mean_abs_ls.append(round(np.mean(curr_df['Abs_connection'].tolist()),2)) 
    mean_rel_ls.append(round(np.mean(curr_df['Rel_connection'].tolist()),2)) 
    std_abs_ls.append(round(np.std(curr_df['Abs_connection'].tolist()),2)) 
    std_rel_ls.append(round(np.std(curr_df['Rel_connection'].tolist()),2))
    CV_abs_ls.append(round(np.std(curr_df['Abs_connection'].tolist())/np.mean(curr_df['Abs_connection'].tolist()),2)) 
    CV_rel_ls.append(round(np.std(curr_df['Rel_connection'].tolist())/np.mean(curr_df['Rel_connection'].tolist()),2))
    p50_abs_ls.append(round(np.percentile(curr_df['Abs_connection'].tolist(),50),2)) 
    p50_rel_ls.append(round(np.percentile(curr_df['Rel_connection'].tolist(),50),2))
    
stats_ranked_df = pd.DataFrame(index=rank_df['Connection_rank'].unique())
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
top_rank_df = df[(df['W']>=desired_count) & (df['rank']<last_input_neuron)].copy()
curr_df = top_rank_df[['rank', 'type_pre', 'instance_post' ]].copy()
curr_df.set_index('instance_post', inplace = True)
curr_df = curr_df.pivot_table(values='rank', index=curr_df.index, columns='type_pre', aggfunc='first').copy()
curr_df.fillna(100, inplace = True)# Penalizing absent nueron with rank=100
rank_column_order = curr_df.median().sort_values().index.tolist()


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

#Neuron filter based on "presence_threshold"
presence_threshold_neuron_filter = thr_rel_presence_absence_df['Presynaptic neuron'].tolist()
presence_threshold_rank_column_order = [neuron for neuron in rank_column_order if neuron in presence_threshold_neuron_filter]


########################################## PRESENCE - ABSENCE of a partner ####################################
###############################################################################################################
# Turning the dataset to binary
binary_df = counting_instances_df.T.copy()
binary_df[binary_df.notnull()] = 1
#binary_df[binary_df.isnull()] = 0 # Fo now, do not excecuto so that values remain NaN

#sorting
column_order = binary_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
binary_sum_sorted_df = binary_df[column_order] # swapping order of columns
binary_rank_sorted_df = binary_df[rank_column_order] # swapping order of columns



##########################################  RELATIVE SYNAPTIC COUNTS ##########################################
###############################################################################################################
#For REALTIVE QUANTIFICATIONS (using 'column_percent')

#Checking the most popular presynaptic partners based on: 

#1) neuron counts across columns
top_rank_popular_neuron_ls = identity_type_df.stack().value_counts().index.tolist()
top_rank_popular_neuron_ls

#2) total percentatge of synaptic count across columns using top-rank neuron data or all data above syn threshold! 
#Synaptic strengh filter
#All data above syn threshold
syn_df = df[df['W']>=desired_count].copy()
syn_type_df = pd.DataFrame(syn_df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe, filtered

#Top-rank neuron data
top_rank_df = df[(df['W']>=desired_count) & (df['rank']<last_input_neuron)].copy()
top_rank_type_df = pd.DataFrame(top_rank_df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe, filtered
type_df = pd.DataFrame(df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe

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


#Taking the most popular of the popular based on descending values of the mean synaptic counts acroos colums
popularity_neuron_based_on_count_percentatge_ls = syn_popularity_rel_df.aggregate('mean', axis = 0).sort_values(ascending =False).index.tolist()
print(popularity_neuron_based_on_count_percentatge_ls[0:last_input_neuron])


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
syn_df = df[df['W']>=desired_count].copy()
syn_type_df = pd.DataFrame(syn_df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe, filtered

#Top-rank neuron data
top_rank_df = df[(df['W']>=desired_count) & (df['rank']<last_input_neuron)].copy()
top_rank_type_df = pd.DataFrame(top_rank_df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe, filtered
type_df = pd.DataFrame(df.groupby(['instance_post', 'type_pre']).agg({'W':sum, 'column_percent':sum})) #Neuron type dataframe

popularity_abs_connections_dict = {}
syn_popularity_abs_connections_dict = {}
top_rank_popularity_abs_connections_dict = {}
for pre in top_rank_popular_neuron_ls: # popular neurons
    
    #Synaptic filter included
    #Top-rank neuron data
    temp_percent_ls = []
    for post in top_rank_type_df.index.levels[0].tolist(): #Columns
        if pre in top_rank_type_df.loc[post].index:
            temp_percent_ls.append(round(top_rank_type_df.loc[post,pre]['W'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    top_rank_popularity_abs_connections_dict[pre] = temp_percent_ls
    
    #Synaptic strengh filter
    temp_percent_ls = []
    for post in syn_type_df.index.levels[0].tolist(): #Columns
        if pre in syn_type_df.loc[post].index:
            temp_percent_ls.append(round(syn_type_df.loc[post,pre]['W'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    syn_popularity_abs_connections_dict[pre] = temp_percent_ls
    
    #No filter
    temp_percent_ls = []
    for post in type_df.index.levels[0].tolist(): #Columns
        if pre in type_df.loc[post].index:
            temp_percent_ls.append(round(type_df.loc[post,pre]['W'],2))
        else:
            temp_percent_ls.append(np.nan)# 0 for non existing pre in post space
            
    popularity_abs_connections_dict[pre] = temp_percent_ls
        
top_rank_popularity_abs_df = pd.DataFrame(top_rank_popularity_abs_connections_dict)
top_rank_popularity_abs_df.index = top_rank_type_df.index.levels[0]

syn_popularity_abs_df = pd.DataFrame(syn_popularity_abs_connections_dict)
syn_popularity_abs_df.index = syn_type_df.index.levels[0]

popularity_abs_df = pd.DataFrame(popularity_abs_connections_dict)
popularity_abs_df.index = type_df.index.levels[0]


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
               columns =['instance_post', 'W'])
        curr_EM_df['fly_ID'] = 'EM'
        curr_EM_df['type_pre'] = partner
        curr_EM_df['type_post'] = neuron_of_interest
        curr_EM_df['column_id'] = [string.split('::')[1] for string in curr_instance_post]
        curr_EM_df.dropna(how='any', axis=0, inplace=True)


        #Concatenating ExM and EM dataframes
        curr_ExM_EM_df = pd.concat([curr_ExM_df,curr_EM_df])
        curr_ExM_EM_df.reset_index(drop=True, inplace=True)
        ExM_EM_absolut_counts[partner] = curr_ExM_EM_df



############################################ CORRELATION MATRICES #############################################
###############################################################################################################

# Calculating statitstical significance for all correlations

#Function:
def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues
 

# Correlation across columns between pair of neurons
# Element-wise pearson correlation. Range: -1 to +1

###Not removing NaNs ( a very problematic case)
curr_df = syn_popularity_rel_df[presence_threshold_rank_column_order].copy() #  filtering based on " presence_threshold"
correlation_rel_df = curr_df.corr(method='pearson', min_periods=1)
#Calculating p_values
p_values_correlation_rel_df = calculate_pvalues(correlation_rel_df) 
p_values_correlation_rel_df_asterix_df = p_values_correlation_rel_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

###Same but replacing NaNs with zeros (The logic thing to do. NaN means actually no connection, so zero is find)
curr_df = curr_df.fillna(0).copy()
correlation_rel_no_NaN_df = curr_df.corr(method='pearson', min_periods=1)
#Calculating p_values
p_values_correlation_rel_no_NaN_df = calculate_pvalues(correlation_rel_no_NaN_df) 
p_values_correlation_rel_no_NaN_df_asterix_df = p_values_correlation_rel_no_NaN_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

###Not removing NaNs ( a very problematic case)
#TODO, >>>> WARNING!!!, check this filter for abs! Is it correct? 
curr_df = syn_popularity_abs_df[presence_threshold_rank_column_order].copy() #  filtering based on " presence_threshold"

correlation_abs_df = curr_df.corr(method='pearson', min_periods=1)
#Calculating p_values
p_values_correlation_abs_df = calculate_pvalues(correlation_abs_df) 
p_values_correlation_abs_df_asterix_df = p_values_correlation_abs_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

###Same but replacing NaNs with zeros (The logic thing to do. NaN means actually no connection, so zero is find)
curr_df = curr_df.fillna(0).copy()
correlation_abs_no_NaN_df = curr_df.corr(method='pearson', min_periods=1)
p_values_correlation_abs_no_NaN_df = calculate_pvalues(correlation_abs_no_NaN_df) 
p_values_correlation_abs_no_NaN_df_asterix_df = p_values_correlation_abs_no_NaN_df.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

# Some sorting based on correlation values
#For relative counts
column_order = correlation_rel_no_NaN_df.sum().sort_values(ascending=False).index.tolist() # new column order based on sum (it will create a gradien from most-correlated to most.anticorrelated)
sorted_correlation_rel_no_NaN_df= correlation_rel_no_NaN_df[column_order] # swpapping columns
sorted_p_values_correlation_rel_no_NaN_df_asterix_df = p_values_correlation_rel_no_NaN_df_asterix_df[column_order]  # swpapping columns

#For absolute counts
column_order = correlation_abs_no_NaN_df.sum().sort_values(ascending=False).index.tolist() # new column order based on sum (it will create a gradien from most-correlated to most.anticorrelated)
sorted_correlation_abs_no_NaN_df= correlation_abs_no_NaN_df[column_order] # swpapping columns
sorted_p_values_correlation_abs_no_NaN_df_asterix_df = p_values_correlation_abs_no_NaN_df_asterix_df[column_order]  # swpapping columns


# Removing the +1 correlated diagonal (setting it to NaN)
#For relative counts
correlation_rel_no_NaN_df.replace(1.0, np.NaN, inplace = True)
sorted_correlation_rel_no_NaN_df.replace(1.0, np.NaN, inplace = True)

#For absolute coutns
correlation_abs_no_NaN_df.replace(1.0, np.NaN, inplace = True)
sorted_correlation_abs_no_NaN_df.replace(1.0, np.NaN, inplace = True)


############################################# COEFICIENT OF VARIATION #########################################
###############################################################################################################

# Table for Coefficient of variation calculations

#Consider filtering some columns (here presynaptic neurons) or indexes that are not interested
curr_rel_stats_df  = syn_popularity_rel_df[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"
curr_abs_stats_df  = syn_popularity_abs_df[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"
#curr_rel_stats_df  = syn_popularity_rel_df.filter(regex='D', axis=0).copy() # Filterinf index base on name
#curr_abs_stats_df  = syn_popularity_abs_df.filter(regex='D', axis=0).copy() # Filterinf index base on name

#Calculate basic statistcs
curr_rel_stats_df = curr_rel_stats_df[curr_rel_stats_df.max().sort_values(ascending = False).index].describe()
curr_abs_stats_df = curr_abs_stats_df[curr_abs_stats_df.max().sort_values(ascending = False).index].describe()

# Calculate coefficient of variation
curr_rel_stats_df.loc['C.V.'] = curr_rel_stats_df.loc['std'] / curr_rel_stats_df.loc['mean']
curr_abs_stats_df.loc['C.V.'] = curr_abs_stats_df.loc['std'] / curr_abs_stats_df.loc['mean']


# Calculate mean of all statistics
curr_rel_stats_df['mean'] = curr_rel_stats_df.mean(axis=1)
curr_abs_stats_df['mean'] = curr_abs_stats_df.mean(axis=1)



########################################### DIMENTIONALITY Reduction ##########################################
###############################################################################################################

#For relative counts:
# Data
rel_data = syn_popularity_rel_df[presence_threshold_rank_column_order].copy()
rel_data= rel_data.fillna(0)
rel_data_array = rel_data.to_numpy(dtype=int,copy=True).T

# PCA
# Standardize

rel_data_array_norm = rel_data_array-rel_data_array.mean(axis=0)
rel_data_array_norm /= rel_data_array_norm.std(axis=0)
n = rel_data_array_norm.shape[0]

# Cov matrix and eigenvectors
rel_cov = (1/n) * rel_data_array_norm @ rel_data_array_norm.T
rel_eigvals, rel_eigvecs = np.linalg.eig(rel_cov)
k = np.argsort(rel_eigvals)[::-1]
rel_eigvals = rel_eigvals[k]
rel_eigvecs = rel_eigvecs[:,k]

#For absolute counts:
# Data
abs_data = syn_popularity_abs_df[presence_threshold_rank_column_order].copy()
abs_data= abs_data.fillna(0)
abs_data_array = abs_data.to_numpy(dtype=int,copy=True).T

# PCA
# Standardize

abs_data_array_norm = abs_data_array-rel_data_array.mean(axis=0)
abs_data_array_norm /= abs_data_array_norm.std(axis=0)
n = abs_data_array_norm.shape[0]

# Cov matrix and eigenvectors
abs_cov = (1/n) * abs_data_array_norm @ abs_data_array_norm.T
abs_eigvals, abs_eigvecs = np.linalg.eig(abs_cov)
k = np.argsort(abs_eigvals)[::-1]
abs_eigvals = abs_eigvals[k]
abs_eigvecs = abs_eigvecs[:,k]



################################################## PCA PLOTS ##################################################
#Plotting Explained variability across dimensions

##Plotting  the square-root eigenvalue spectrum
fig,axs = plt.subplots(nrows =1, ncols = 1)
axs.plot(rel_eigvals/np.sum(rel_eigvals) * 100,'-o',color='black')
axs.set_title(f'{neuron_of_interest}, Eigenvalue spectrum, syn>={desired_count}')
axs.set_xlabel('Dimensions')
axs.set_ylabel('Explained-variability (percentage)')

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\PCA-exlpained-variability-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: PCA explained variability plotted and saved')
plt.close(fig)


##Plotting PCA1 vs PCA2
fig,axs = plt.subplots(nrows =1, ncols = 1)
axs.scatter(rel_data_array.T @ rel_eigvecs[:,0],rel_data_array.T @ rel_eigvecs[:,1])
axs.set_title(f'{neuron_of_interest}, PCA1 vs PCA2, syn>={desired_count}')
axs.set_xlabel('PCA1')
axs.set_ylabel('PCA2')

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\PCA1-PCA2-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: PCA1 vs PCA2 plotted and saved')
plt.close(fig)

##Plotting contribution of neurons to PCA
fig,axs = plt.subplots(nrows =1, ncols = 1)
im = axs.imshow(np.array([rel_eigvecs[:,0],rel_eigvecs[:,1]]).T,cmap='coolwarm',aspect='auto')
plt.colorbar(im, ax=axs)
axs.set_xlabel('Principal components (PCs)')
ax = plt.gca()
a = list(range(0, rel_eigvecs.shape[0]))
axs.set_yticks(a)
axs.set_yticklabels(rel_data.columns)
plt.title('Contribution of neurons to PCs')

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Contribution-neurons-to-PCA-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Contribution of neurons to PCA plotted and saved')
plt.close(fig)

#%% 
############################################# PLOTTING SECTION ##############################################
#############################################################################################################
#############################################################################################################




################################################# BAR - PLOTS ##############################################
############################################################################################################
# Bar plot showing presence and absence of neuron partners

#Figure
fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (40*cm, 15*cm))
fig.tight_layout(pad=10) # Adding some space between subplots

color_absent = [204/255,236/255,230/255]
color_present = [27/255,168/255,119/255]
color_present = sns.color_palette("light:#5A9", as_cmap=False)[-1]


# First axis
sorted_abs_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                    edgecolor = None, ax = axs[0],legend=False)
axs[0].set_title(f'{neuron_of_interest}, Presence / absence across columns, syn>={desired_count}')
#axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
axs[0].set_xlabel('Presynaptic neuron')
axs[0].set_ylabel('Number of columns')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)


# Next axis
sorted_rel_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                    edgecolor = None, ax = axs[1],legend=False)
axs[1].set_title(f'{neuron_of_interest}, Presence / absence across columns, syn>={desired_count}')
#axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
axs[1].set_xlabel('Presynaptic neuron')
axs[1].set_ylabel('% of columns')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# Next axis
sorted_thr_rel_presence_absence_df.set_index('Presynaptic neuron').plot(kind='bar', stacked=True, color=[color_present, color_absent], 
                                                                    edgecolor = "black", ax = axs[2],legend=False)
axs[2].set_title(f'{neuron_of_interest}, Presence / absence across columns above {presence_threshold}, syn>={desired_count}')
#axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
axs[2].set_xlabel('Presynaptic neuron')
axs[2].set_ylabel('% of columns')
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Percentage_columns_partner_presence_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Percentatge across columns plotted and saved')
plt.close(fig)


#%% 
############################################### HEATMAP - PLOTS ############################################
############################################################################################################

################################################ BINARY PLOTS ##############################################
## Visualizing the existance of a presynaptic neuron
#Heatmap plots

#Data filtering
_data =binary_rank_sorted_df[presence_threshold_neuron_filter] # filtering based on " presence_threshold"

#Figure
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
_palette = sns.color_palette("light:#5A9", as_cmap=True)

heatmap = sns.heatmap(cmap =_palette, data = _data, vmin=0, vmax=1, linewidths=0.2,
                linecolor='k', cbar=False, ax = axs, square=True) 
axs.set_title(f'{neuron_of_interest}, Binary: presence - absence, rank sorted, syn>={desired_count}')
axs.set_ylabel('Column')
axs.set_xlabel('Presynaptic neuron')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "binary_rank_sorted_df"
axs.set_yticks(range(len(_data.index)))
axs.set_yticklabels(_data.index)

if save_figures:
    # Quick plot saving
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Binary-heatmap-presence-absence-partner_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Binary heatmap plotted and saved')
plt.close(fig)


################################################ INSTANCE COUNTS ##############################################
## Visualizing instance (copies of the same neuron type) counts
#Heatmap plots

#Specifi color settings for instances:

color_palette_name = "tab10" # "magma", "rocket", "tab10", "plasma", , "viridis", "flare"

#Data filtering
curr_df = counting_instances_df.T[presence_threshold_neuron_filter].copy() #  filtering based on " presence_threshold"

#Sorting based on max, sum and count
column_order = curr_df.max().sort_values(ascending=False).index.tolist() # Sorting based on MAX of all values in the column
curr_max_sorted_df = curr_df[column_order] # swapping order of columns
column_order = curr_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
curr_sum_sorted_df = curr_df[column_order] # swapping order of columns
column_order = curr_df.count().sort_values(ascending=False).index.tolist() # Sorting based on COUNT of all values in the column
curr_count_sorted_df = curr_df[column_order] # swapping order of columns

#Sorting based on rank
curr_rank_sorted_df = counting_instances_df.T[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"
curr_rank_sorted_df[presence_threshold_neuron_filter]


# Data
_data =curr_rank_sorted_df 

#Figure
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
max_count = int(max(counting_instances_df.max()))
_palette = sns.color_palette(color_palette_name,max_count)

#Plot
heatmap = sns.heatmap(cmap=_palette, data=_data, vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, Instance count, sorted by rank, syn>={desired_count}')
axs.set_ylabel('Columns')
axs.set_xlabel('Presynaptic neurons')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.index)))
axs.set_yticklabels(_data.index)

# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.columns)))
axs.set_xticklabels(_data.columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Presynaptic-instance-count-per-column-sorted_{dataset_name}_{neuron_of_interest}-vertical.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of instance counts plotted vertically and saved')
plt.close(fig)



#Figure
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
max_count = int(max(counting_instances_df.max()))
_palette = sns.color_palette(color_palette_name, max_count)

# Plot (rotated 90 degrees)
heatmap = sns.heatmap(cmap=_palette, data=_data.transpose(), vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, Instance count, sorted by rank, syn>={desired_count}')
axs.set_xlabel('Columns')
axs.set_ylabel('Presynaptic neurons')

# Reduce font size of x-axis tick labels
for tick_label in heatmap.get_xticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.transpose().index)))
axs.set_yticklabels(_data.transpose().index)

# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.transpose().columns)))
axs.set_xticklabels(_data.transpose().columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Presynaptic-instance-count-per-column-sorted_{dataset_name}_{neuron_of_interest}-horizontal.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of instance counts plotted horizontally and saved')
plt.close(fig)


################################################ RELATIVE  COUNTS  ##############################################
# Visualization of presynaptic contact percentatge for all columns
#Heatmap of presynaptic partners  colorcoded by relative synaptic count


#Data
_data = top_rank_popularity_rel_df[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"


#Figure
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
_palette = sns.color_palette("gist_ncar",n_colors=20)

#First axis

sns.heatmap(cmap = _palette, vmin=0, vmax=50, data = _data, cbar_kws={"shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, count %, rank-sorted (syn>={desired_count})')
axs.set_ylabel('Column')
#axs.set_yticklabels(id_column)
axs.set_xlabel('Presynaptic neuron')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.index)))
axs.set_yticklabels(_data.index)


# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.columns)))
axs.set_xticklabels(_data.columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Relative-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-vertical.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of relative counts plotted vertically and saved')
plt.close(fig)




#Figure
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
_palette = sns.color_palette("gist_ncar",n_colors=20)

#First axis

# Plot (rotated 90 degrees)
sns.heatmap(cmap = _palette, vmin=0, vmax=50, data = _data.transpose(), cbar_kws={"shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, count %, rank-sorted (syn>={desired_count})')
axs.set_xlabel('Column')
#axs.set_yticklabels(id_column)
axs.set_ylabel('Presynaptic neuron')

# Reducing font size of x-axis tick labels
for tick_label in heatmap.get_xticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.transpose().index)))
axs.set_yticklabels(_data.transpose().index)

# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.transpose().columns)))
axs.set_xticklabels(_data.transpose().columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Relative-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-horizontal.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of relative counts plotted horizontally and saved')
plt.close(fig)



################################################### ABSOLUTE COUNTS  #################################################
# Visualization of presynaptic contact percentatge for all columns
#Heatmap of presynaptic partners  colorcoded by relative synaptic count


#Data
_data = top_rank_popularity_abs_df[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"


#Figure
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
_palette = sns.color_palette("gist_ncar",n_colors=20)

#First axis

sns.heatmap(cmap = _palette, vmin=0, vmax=50, data = _data, cbar_kws={"shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, absolute count, rank-sorted (syn>={desired_count})')
axs.set_ylabel('Column')
#axs.set_yticklabels(id_column)
axs.set_xlabel('Presynaptic neuron')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.index)))
axs.set_yticklabels(_data.index)


# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.columns)))
axs.set_xticklabels(_data.columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Absolute-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-vertical.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of absolute counts plotted vertically and saved')
plt.close(fig)


#Figure
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
_palette = sns.color_palette("gist_ncar",n_colors=20)

#First axis

# Plot (rotated 90 degrees)
sns.heatmap(cmap = _palette, vmin=0, vmax=50, data = _data.transpose(), cbar_kws={"shrink": 0.25}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, absolute count, rank-sorted (syn>={desired_count})')
axs.set_xlabel('Column')
#axs.set_yticklabels(id_column)
axs.set_ylabel('Presynaptic neuron')

# Reducing font size of x-axis tick labels
for tick_label in heatmap.get_xticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.transpose().index)))
axs.set_yticklabels(_data.transpose().index)

# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.transpose().columns)))
axs.set_xticklabels(_data.transpose().columns)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Absolute-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-horizontal.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of absolute counts plotted horizontally and saved')
plt.close(fig)



################################################### NEURONAL RANK  #################################################
# Visualization neuron´s ranks
# Heatmap of presynaptic partners colorcoded by rank


#Data
curr_df = top_rank_df[['rank', 'type_pre', 'instance_post' ]].copy()
curr_df.set_index('instance_post', inplace = True)
curr_df = curr_df.pivot_table(values='rank', index=curr_df.index, columns='type_pre', aggfunc='first').copy()
_data = curr_df[presence_threshold_rank_column_order].copy()#  filtering based on " presence_threshold"



#Figure 
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
total_num_ranks = top_rank_df['rank'].max() + 1
_palette = sns.color_palette("tab20",n_colors=total_num_ranks)

#First axis
sns.heatmap(cmap = _palette, vmin=0, vmax=total_num_ranks, cbar_kws={"ticks":list(range(1,top_rank_df['rank'].max()+2,1)),"shrink": 0.25}, data = _data, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, RANK neurons (syn>={desired_count})')
axs.set_ylabel('Column')
#axes.set_yticklabels(id_column)
axs.set_xlabel('Presynaptic neuron')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.index)))
axs.set_yticklabels(_data.index)


# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.columns)))
axs.set_xticklabels(_data.columns)

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Rank-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-vertical.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of ranks plotted vertically and saved')

plt.close(fig)



#Figure 
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
#_palette = sns.color_palette("rocket",n_colors=20)
total_num_ranks = top_rank_df['rank'].max() + 1
_palette = sns.color_palette("tab20",n_colors=total_num_ranks)

#First axis
sns.heatmap(cmap = _palette, vmin=0, vmax=total_num_ranks, cbar_kws={"ticks":list(range(1,top_rank_df['rank'].max()+2,1)),"shrink": 0.25}, data = _data.transpose(), ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, RANK neurons (syn>={desired_count})')
axs.set_xlabel('Column')
#axes.set_yticklabels(id_column)
axs.set_ylabel('Presynaptic neuron')

# Reducing font size of x-axis tick labels
for tick_label in heatmap.get_xticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "_data"
axs.set_yticks(range(len(_data.transpose().index)))
axs.set_yticklabels(_data.transpose().index)

# Add ticks in the X-axis for each row in "_data"
axs.set_xticks(range(len(_data.transpose().columns)))
axs.set_xticklabels(_data.transpose().columns)

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Rank-connectivity-heatmap-across-columns_{dataset_name}_{neuron_of_interest}-horizontal.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of ranks plotted horizontally and saved')
plt.close(fig)



############################################ CORRELATION MATRIXES  ##########################################
# Visualization of Hieracrchical clustering
# Heatmap of presynaptic partners' person correlation
# Relative numbers


correlation_rel_no_NaN_df.replace(np.NaN,1.0, inplace = True)
_palette = sns.color_palette("vlag", as_cmap=True) # Diverging palette
g = sns.clustermap(cmap = _palette, data = correlation_rel_no_NaN_df, annot = np.array(p_values_correlation_rel_no_NaN_df_asterix_df), fmt='', annot_kws={"size":16, "color": "k"})

g.fig.suptitle(f'{neuron_of_interest} partners, pearson correlation matrix, hierarchical clustering, % of count(syn>={desired_count})') 
g.ax_heatmap.set_xlabel('Presynaptic neuron')
g.ax_heatmap.set_ylabel('Presynaptic neuron')
g.fig.subplots_adjust(top=0.9)
x0, y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([x0+1, y0, g.ax_cbar.get_position().width/5, g.ax_cbar.get_position().width])
g.ax_cbar.set_title('pearson')


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Hierarchical-clustering-correlation-relative-counts_{dataset_name}_{neuron_of_interest}.pdf'
    g.savefig(save_path+figure_title)
    print('FIGURE: Visualization of pearson correlation and hierarchical clustering plotted and saved')
plt.close(fig)


correlation_abs_no_NaN_df.replace(np.NaN,1.0, inplace = True)
_palette = sns.color_palette("vlag", as_cmap=True) # Diverging palette
g = sns.clustermap(cmap = _palette, data = correlation_rel_no_NaN_df, annot = np.array(p_values_correlation_rel_no_NaN_df_asterix_df), fmt='', annot_kws={"size":16, "color": "k"})

g.fig.suptitle(f'{neuron_of_interest} partners, pearson correlation matrix, hierarchical clustering, absolute count(syn>={desired_count})') 
g.ax_heatmap.set_xlabel('Presynaptic neuron')
g.ax_heatmap.set_ylabel('Presynaptic neuron')
g.fig.subplots_adjust(top=0.9)
x0, y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([x0+1, y0, g.ax_cbar.get_position().width/5, g.ax_cbar.get_position().width])
g.ax_cbar.set_title('pearson')


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Hierarchical-clustering-correlation-absolute-counts_{dataset_name}_{neuron_of_interest}.pdf'
    g.savefig(save_path+figure_title)
    print('FIGURE: Visualization of pearson correlation and hierarchical clustering plotted and saved')
plt.close(fig)


############################################# BOXPLOTS - PLOTS ##############################################
#############################################################################################################

# Plotting box plots of presynaptic counts
# Relative and absolute counts across columns

# Data
_data = syn_popularity_rel_df.copy()[presence_threshold_rank_column_order]#  filtering based on " presence_threshold"

#Figure
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(30*cm, 30*cm))
fig.tight_layout(pad=10) # Adding some space between subplots


# First axes 

sns.boxplot(data = _data[_data.max().sort_values(ascending = False).index], ax = axs[0]) 
axs[0].set_title(f'{neuron_of_interest},  count % of popular neurons (syn>={desired_count})')
axs[0].set_ylabel('Synaptic count (%) ', size = 12)
axs[0].set_xlabel('Presynaptic neuron', size = 12)
axs[0].set_xticklabels(_data[_data.max().sort_values(ascending = False).index], rotation=90, size = 10)
axs[0].set_yticklabels(axs[0].get_yticks(), size = 8)


# Data
_data = syn_popularity_abs_df.copy()[presence_threshold_rank_column_order]#  filtering based on " presence_threshold"

# Next axes 
sns.boxplot(data = _data[_data.max().sort_values(ascending = False).index], ax = axs[1]) 
axs[1].set_title(f'{neuron_of_interest},  Absolute count of popular neurons (syn>={desired_count})')
axs[1].set_ylabel('Synaptic count', size = 12)
axs[1].set_xlabel('Presynaptic neuron', size = 12)
axs[1].set_xticklabels(_data[_data.max().sort_values(ascending = False).index], rotation=90, size = 10)
axs[1].set_yticklabels(axs[1].get_yticks(), size = 8)


#Plot saving
if save_figures:
    save_path =  r'E:\Connectomics-Data\FlyWire\Pdf-plots' #r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Box-plot-presynaptic-partners_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of presynaptic partners contacts')
plt.close(fig)



################################################ BAR - PLOTS ################################################
#############################################################################################################

# Plotting bar plots of presynaptic counts
# Quick plot across neurons of basic descriptive statistics of variability

#Figure
fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(15*cm, 30*cm))
fig.tight_layout(pad=8) # Adding some space between subplots

#Data
_data = curr_rel_stats_df.copy() 


#First axis
sns.barplot(data = _data.iloc[[2]], ax = axs[0] )
axs[0].axhline(y = _data.iloc[[2]]['mean'][0], color = 'k', linestyle = 'dashed')  
axs[0].set_title(f'{neuron_of_interest} partners, variability measure: std, % of count(syn>={desired_count})')
axs[0].set_ylabel(_data.index[2], size = 10)
axs[0].set_xlabel(f'Presynaptic neuron', size = 10)
axs[0].set_xticklabels(_data.iloc[[2]], rotation=90)
axs[0].set_yticklabels(axs[1].get_yticks().round(1), size = 6)
axs[0].set_xticklabels(_data.columns.tolist(), size = 8)



#First axis
sns.barplot(data = _data.iloc[[-1]], ax = axs[1] )
axs[1].axhline(y = _data.iloc[[-1]]['mean'][0], color = 'k', linestyle = 'dashed')  
axs[1].set_title(f'{neuron_of_interest} partners, variability measure: C.V, % of count(syn>={desired_count})')
axs[1].set_ylabel(_data.index[-1], size = 10)
axs[1].set_xlabel(f'Presynaptic neuron', size = 10)
axs[1].set_xticklabels(_data.iloc[[-1]], rotation=90)
axs[1].set_yticklabels(axs[1].get_yticks().round(1), size = 6)
axs[1].set_xticklabels(_data.columns.tolist(), size = 8)


#Data
_data = curr_abs_stats_df.copy() 

#Next axis
sns.barplot(data = _data.iloc[[2]], ax = axs[2] )
axs[2].axhline(y = _data.iloc[[2]]['mean'][0], color = 'k', linestyle = 'dashed')  
axs[2].set_title(f'{neuron_of_interest} partners, variability measure: std, absolute count(syn>={desired_count})')
axs[2].set_ylabel(_data.index[2], size = 10)
axs[2].set_xlabel(f'Presynaptic neuron', size = 10)
axs[2].set_xticklabels(_data.iloc[[2]], rotation=90)
axs[2].set_yticklabels(axs[1].get_yticks().round(1), size = 6)
axs[2].set_xticklabels(_data.columns.tolist(), size = 8)

#Next axis
sns.barplot(data = _data.iloc[[-1]], ax = axs[3] )
axs[3].axhline(y = _data.iloc[[-1]]['mean'][0], color = 'k', linestyle = 'dashed')  
axs[3].set_title(f'{neuron_of_interest} partners, variability measure: C.V, absolute count(syn>={desired_count})')
axs[3].set_ylabel(_data.index[-1], size = 10)
axs[3].set_xlabel(f'Presynaptic neuron', size = 10)
axs[3].set_xticklabels(_data.iloc[[-1]], rotation=90)
axs[3].set_yticklabels(axs[1].get_yticks().round(1), size = 6)
axs[3].set_xticklabels(_data.columns.tolist(), size = 8)


#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\Variability-measures_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of variability measures')
plt.close(fig)



################################################ HISTOGRAMS - PLOTS #########################################
#############################################################################################################
# Visualization across data sets


# Plotting ExM and Em data together for following neurons
partner_list = list(ExM_EM_absolut_counts.keys())
binwidth_list = [2,1,4]

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
        data=_data, x="W", hue="fly_ID", multiple="dodge", shrink=1,
        log_scale=False, element="bars", fill=True, binwidth=binwidth_list[i],
        cumulative=False, stat="percent", common_norm=False, legend=False,
        ax=axs[i])

    axs[i].set_title(f'{partner}>{neuron_of_interest}, Column counts histogram ExM and EM, syn>={desired_count}')
    axs[i].set_ylabel('Column count (percent)', size=10)
    axs[i].set_xlabel('Presynaptic contacts', size=10)
    axs[i].tick_params(axis='x', labelsize=8)
    axs[i].tick_params(axis='y', labelsize=8)

    unique_labels = _data['fly_ID'].unique()
    handles = [plt.Rectangle((0, 0), 1, 1, color=sns.color_palette()[i]) for i in range(len(unique_labels))]

    axs[i].legend(handles, unique_labels, frameon=False, fontsize=4)

plt.tight_layout()
plt.show()

#Plot saving
if save_figures:
    save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
    figure_title = f'\ExM-EM-histograms_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of ExM and EM histograms plotted and saved')
plt.close(fig)



























#%% 
############################################### NEUROPIL - PLOTS ############################################
#############################################################################################################

################################################ BINARY COUNTs ###############################################



#Gettting the center point in specific neuropile from database
xyz_neuropil = 'XYZ-ME'
xyz_df = database_df[database_df['seg_id'].isin(root_ids)].copy()
xyz_pre = xyz_df[xyz_neuropil].tolist()
# Split each string by comma and convert the elements to floats
xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])

#Getting list for dot sizes and colors based on instance counts of a pre_partner
pre_partner = 'Dm12'

#Dot sizes
dot_sizes = binary_rank_sorted_df[pre_partner].fillna(0).tolist()
dot_sizes_ME = [size*20 for size in dot_sizes]  # Increase size by a factor of 20
dot_sizes_LO = [size*10 for size in dot_sizes]  # Increase size by a factor of 10

OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME,c='#458A7C')  # Adjust the size (s) as desired
navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')
ax.azim = -18
ax.elev = -148
plt.show()

#Plot saving
if save_figures:
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Meshes_XYZ_positions_ME_binary_{dataset_name}_{pre_partner}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of XYZ positions plotted and saved')
plt.close(fig)


################################# PLOTTING EACH  NEURON IN DIFFERENT PDF PAGES ###########################################


from matplotlib.backends.backend_pdf import PdfPages
# Assuming pre_partner_list is a list of objects to be plotted
pre_partner_list = binary_rank_sorted_df.columns.tolist()
OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']


# Create a PDF file to save the plots
save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
figure_title = f'\Meshes_XYZ_positions_ME_binary_{dataset_name}_{neuron_of_interest}_many_pages.pdf'
outputPath =  save_path+figure_title
pdf_pages = PdfPages(outputPath)

# Loop through the objects and create subplots
for i, pre_partner in enumerate(pre_partner_list):
    # Generate the plot for the current object
    dot_sizes = binary_rank_sorted_df[pre_partner].fillna(0).tolist()
    dot_sizes_ME = [size*20 for size in dot_sizes]  # Increase size by a factor of 20
    dot_sizes_LO = [size*10 for size in dot_sizes]  # Increase size by a factor of 10

    fig = plt.figure(figsize=(8.27, 11.69))  # DIN4 page size (in inches)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the object
    ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME, c='#458A7C', alpha=0.9)
    navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')

    #Rotating the view
    ax.azim = -18
    ax.elev = -148

    # Set plot title
    ax.set_title(f"Presynaptic partner: {pre_partner}")

    # Add any additional customization to the plot

    # Save the current plot as a subplot in the PDF file
    pdf_pages.savefig(fig, bbox_inches='tight')

    # Close the current figure
    plt.close(fig)

# Save and close the PDF file
pdf_pages.close()

print(f"Plots saved in {outputPath}")



################################# PLOTTING EACH  NEURON IN SAME PDF PAGE ###########################################


from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

# Assuming pre_partner_list is a list of objects to be plotted
pre_partner_list = binary_rank_sorted_df.columns.tolist()
OL_R = flywire.get_neuropil_volumes(['ME_R'])

# Create a PDF file to save the plots
save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
figure_title = f'\Meshes_XYZ_positions_ME_binary_{dataset_name}_{neuron_of_interest}.pdf'
outputPath =  save_path + figure_title
outputPath = r'E:\Connectomics-Data\FlyWire\Pdf-plots\Test_02_binary.pdf'
pdf_pages = PdfPages(outputPath)
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
    dot_sizes = binary_rank_sorted_df[pre_partner].fillna(0).tolist()
    dot_sizes_ME = [size * 5 for size in dot_sizes]  # Increase size by a factor of X

    # Plot the object
    ax.scatter(
        xyz_pre_arr_new[:, 0],
        xyz_pre_arr_new[:, 1],
        xyz_pre_arr_new[:, 2],
        s=dot_sizes_ME,
        c='#458A7C',
        alpha=0.9
    )
    navis.plot2d([xyz_pre_arr_new, OL_R], method='3d_complex', ax=ax, view=(172, 51)) # scalebar='20 um'

    # Rotating the view
    ax.azim = -18
    ax.elev = -148

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

#Gettting the center point in specific neuropile from database
xyz_neuropil = 'XYZ-ME'
xyz_df = database_df[database_df['seg_id'].isin(root_ids)].copy()
xyz_pre = xyz_df[xyz_neuropil].tolist()
# Split each string by comma and convert the elements to floats
xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])

#Getting list for dot sizes and colors based on instance counts of a pre_partner
pre_partner = 'Dm12'

#Dot sizes
dot_sizes = counting_instances_df.T[pre_partner].fillna(0).tolist()
dot_sizes_ME = [size*20 for size in dot_sizes]  # Increase size by a factor of 20
dot_sizes_LO = [size*10 for size in dot_sizes]  # Increase size by a factor of 10

size_color_map = {}
color_palette = sns.color_palette(color_palette_name, len(set(dot_sizes)))

#Dot colors
dot_colors = []

for size in dot_sizes:
    if size != 0.0 and size not in size_color_map:
        size_color_map[size] = color_palette[len(size_color_map)]
    #dot_colors.append(size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0))
    color = size_color_map.get(size, (1.0, 1.0, 1.0)) if size != 0.0 else (1.0, 1.0, 1.0)
    color = (*color[:3], 1.0)  # Make color fully opaque
    dot_colors.append(color)




OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_ME,c=dot_colors)  # Adjust the size (s) as desired
navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')
ax.azim = -18
ax.elev = -148
plt.show()

#Plot saving
if save_figures:
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Meshes_XYZ_positions_ME_{dataset_name}_{pre_partner}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of XYZ positions plotted and saved')
plt.close(fig)

#Gettting the center point in specific neuropile from database
xyz_neuropil = 'XYZ-LO'
xyz_df = database_df[database_df['seg_id'].isin(root_ids)].copy()
xyz_pre = xyz_df[xyz_neuropil].tolist()
# Split each string by comma and convert the elements to floats
xyz_pre_arr = np.array([list(map(float, s.split(','))) for s in xyz_pre])
xyz_pre_arr_new = xyz_pre_arr * np.array([4,4,40])

OL_R = flywire.get_neuropil_volumes(['LO_R']) #['ME_R','LO_R','LOP_R']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_pre_arr_new[:, 0], xyz_pre_arr_new[:, 1], xyz_pre_arr_new[:, 2], s=dot_sizes_LO,c=dot_colors)  # Adjust the size (s) as desired
navis.plot2d([xyz_pre_arr_new,OL_R], method='3d_complex', ax=ax,view=(172, 51),scalebar = '20 um')
ax.azim = -6
ax.elev = -57
plt.show()

#Plot saving
if save_figures:
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Meshes_XYZ_positions_LO_{dataset_name}_{pre_partner}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of XYZ positions plotted and saved')
plt.close(fig)





##############################################################################################################
##############################################################################################################
##############################################################################################################

#Temp CODE, Debugging

# tem_df = pd.read_csv(r'E:\Connectomics-Data\FlyWire\Excels\optic_lobe_ids_left.csv')
# ol_id_list = tem_df.columns.tolist()

# def non_match_elements(list_a, list_b):
#     non_match = []
#     for i in list_a:
#         if i not in list_b:
#             non_match.append(i)
#     return non_match


# non_match = non_match_elements(ol_id_list, id_column,)
# print("No match elements: ", non_match)

# %%
# # OLD CODE


# ########################################### DIMENTIONALITY Reduction ##########################################
# ###############################################################################################################
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# ### PCA across columns - relative counts

# # Define number of clusters (not mandatory for PCA, just for exploration)
# num_clusters = 2

# #Defining range of neurons (featured) to plot based on rank (defining the subset for the PCA)
# features = presence_threshold_rank_column_order # filtering based on " presence_threshold"

# #Get the features data
# curr_rel_df = syn_popularity_rel_df[presence_threshold_rank_column_order].copy()
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
# features = presence_threshold_rank_column_order # filtering based on " presence_threshold"

# #Get the features data
# curr_abs_df = syn_popularity_abs_df[presence_threshold_rank_column_order].copy()
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



# ################################################## PCA PLOTS ##################################################
# #Plotting PCA plots

# #Figure
# fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(30*cm, 30*cm)) #figsize=(20*cm, 40*cm)), figsize=(40*cm, 80*cm))
# fig.tight_layout(pad=8) # Adding some space between subplots

# #First Axis
# sns.scatterplot(x="pca1", y="pca2", data=results_rel_columns, ax = axs[0])
# axs[0].set_title(f'{neuron_of_interest}, PCA - count % of popular neurons (syn>={desired_count})')
# axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# #Next Axis
# sns.scatterplot(x="pca1", y="pca2", data=results_abs_columns, ax = axs[1])
# axs[1].set_title(f'{neuron_of_interest}, PCA - absolute count of popular neurons (syn>={desired_count})')
# axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)



# #Plot saving
# if save_figures:
#     save_path = r'E:\Connectomics-Data\FlyWire\Pdf-plots'
#     figure_title = f'\PCA-relative-subset-connectivity_{dataset_name}_{neuron_of_interest}.pdf'
#     fig.savefig(save_path+figure_title)
#     print('FIGURE: PCA plotted and saved')
# plt.close(fig)

