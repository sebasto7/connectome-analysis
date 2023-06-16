# -*- coding: utf-8 -*-
"""
Created on Mon June 12 14:43:16 2023

@author: smolina

variability analysis of presynaptic inputs
"""

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
fileDate = '20230615'
fileName = f'Tm9_neurons_input_count_ME_L_{fileDate}.xlsx'
fileName_database = f'Tm9 proofreadings_{fileDate}.xlsx'


#%% 
################################################## PRE-ANALYSIS ###############################################
###############################################################################################################

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
################################################## ANALYSIS ###################################################
###############################################################################################################

# Additional useful data frames. 

############################################# IDENTITY AND CONNECTIONS ########################################

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
    

####################################################### RANKS #########################################################
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


##################################################### INSTANCES OF NEURONS ###############################################
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

#Sorting dataframe
sorted_abs_presence_absence_df = abs_presence_absence_df.sort_values(by=['Present'], ascending=False)
sorted_rel_presence_absence_df = rel_presence_absence_df.sort_values(by=['Present'], ascending=False)


####################################### PRESENCE - ABSENCE of a partner ###################################
# Turning the dataset to binary
binary_df = counting_instances_df.T.copy()
binary_df[binary_df.notnull()] = 1
#binary_df[binary_df.isnull()] = 0 # Fo now, do not excecuto so that values remain NaN

#sorting
column_order = binary_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
binary_sum_sorted_df = binary_df[column_order] # swapping order of columns
binary_rank_sorted_df = binary_df[rank_column_order] # swapping order of columns

#%% 
############################################### HEATMAP - PLOTS ############################################
############################################################################################################

################################################ BINARY PLOTS ##############################################
## Visualizing the existance of a presynaptic neuron
#Heatmap plots


fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
_palette = sns.color_palette("light:#5A9", as_cmap=True)

heatmap = sns.heatmap(cmap =_palette, data = binary_rank_sorted_df, vmin=0, vmax=1, linewidths=0.2,
                linecolor='k', cbar=False, ax = axs, square=True) 
axs.set_title(f'{neuron_of_interest}, Binary: presence - absence, rank sorted, syn>={desired_count}')
axs.set_ylabel('Column')
axs.set_xlabel('Presynaptic neuron')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "binary_rank_sorted_df"
axs.set_yticks(range(len(binary_rank_sorted_df.index)))
axs.set_yticklabels(binary_rank_sorted_df.index)

if save_figures:
    # Quick plot saving
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Binary-heatmap-presence-absence-partner_{dataset_name}_{neuron_of_interest}.pdf'
    fig.savefig(save_path+figure_title)
plt.close(fig)


################################################ INSTANCE COUNTS ##############################################
## Visualizing instance (copies of the same neuron type) counts

#Specifi color settings for instances:

color_palette_name = "tab10" # "magma", "rocket", "tab10", "plasma", , "viridis", "flare", 
#Heatmap plots

#Sorting
curr_df = counting_instances_df.T.copy()
column_order = curr_df.max().sort_values(ascending=False).index.tolist() # Sorting based on MAX of all values in the column
curr_max_sorted_df = curr_df[column_order] # swapping order of columns
column_order = curr_df.sum().sort_values(ascending=False).index.tolist() # Sorting based on SUM of all values in the column
curr_sum_sorted_df = curr_df[column_order] # swapping order of columns
column_order = curr_df.count().sort_values(ascending=False).index.tolist() # Sorting based on COUNT of all values in the column
curr_count_sorted_df = curr_df[column_order] # swapping order of columns
#Rank sorting
curr_rank_sorted_df = counting_instances_df.T.copy()[rank_column_order]

#Figure
fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(10*cm, 20*cm))
max_count = int(max(counting_instances_df.max()))
_palette = sns.color_palette(color_palette_name,max_count)

#Plot
heatmap = sns.heatmap(cmap=_palette, data=curr_rank_sorted_df, vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.5}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, Instance count, sorted by rank, syn>={desired_count}')
axs.set_ylabel('Columns')
axs.set_xlabel('Presynaptic neurons')

# Reducing font size of y-axis tick labels
for tick_label in heatmap.get_yticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

# Add ticks in the Y-axis for each row in "curr_rank_sorted_df"
axs.set_yticks(range(len(curr_rank_sorted_df.index)))
axs.set_yticklabels(curr_rank_sorted_df.index)


#Plot saving
if save_figures:
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Presynaptic-instance-count-per-column-sorted_{dataset_name}_{neuron_of_interest}-horizontal.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of instance counts plotted and saved')
plt.close(fig)


fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(30*cm, 15*cm))
max_count = int(max(counting_instances_df.max()))
_palette = sns.color_palette(color_palette_name, max_count)

# Plot (rotated 90 degrees)
heatmap = sns.heatmap(cmap=_palette, data=curr_rank_sorted_df.transpose(), vmin=1, vmax=max_count+1, cbar_kws={"ticks": list(range(1, max_count+1, 1)), "shrink": 0.5}, ax=axs, square=True)
axs.set_title(f'{neuron_of_interest}, Instance count, sorted by rank, syn>={desired_count}')
axs.set_xlabel('Columns')
axs.set_ylabel('Presynaptic neurons')

# Reduce font size of x-axis tick labels
for tick_label in heatmap.get_xticklabels():
    tick_label.set_fontsize(tick_label.get_fontsize() * 0.5)

#Plot saving
if save_figures:
    save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots' # r'C:\Users\sebas\Documents\Connectomics-Data\FlyWire\Pdf-plots' 
    figure_title = f'\Presynaptic-instance-count-per-column-sorted_{dataset_name}_{neuron_of_interest}-vertical.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of instance counts plotted and saved')
plt.close(fig)

#%% 
############################################### NEUROPIL - PLOTS ############################################
#############################################################################################################

################################################ BINARY COUNTs ###############################################

#TODO Do a loop for plotting all presynaptic neurons in same plot page

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


print('Coding here')




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
