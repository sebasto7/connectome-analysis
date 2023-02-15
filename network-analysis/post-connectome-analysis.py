# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:32:50 2021

@author: Sebastian Molina-Obando
"""

#%% Importing packages
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from core_functions_network_analysis import path_length_transformation_plot, graph_plot,node_to_node_graph_analysis_and_plot, heatmap_plot
from core_functions_network_analysis import  input_output_plot,direct_indirect_connections_plot, connections_histomgram_plot,centrality_plot



#%% Users parameters
main_data_folder = r'D:\Connectomics-Data'
graph = 'Fib25_data_7medulla_columns'
pkl_file_list = ['Home-column.pickle','A-column.pickle','B-column.pickle','C-column.pickle', 'D-column.pickle', 'E-column.pickle', 'F-column.pickle' ] # Fill in
short_col_names = ['Ho','A','B','C', 'D', 'E', 'F' ] # Fill in


#%% Auto creation of important paths
main_processed_data_folder = os.path.join(main_data_folder,'processed-data',graph)
pkl_folder = main_processed_data_folder

user_parameters = {}
user_parameters['dataPath']= main_data_folder
user_parameters['processed_data_folder']= main_processed_data_folder
user_parameters['graph']= graph # For plot titles
user_parameters['files'] = pkl_file_list
user_parameters['column'] = f'{len(pkl_file_list)} columns analyzed' # For plot titles



save_data = 1
save_figures = 1
_fontsize = 6
node_to_node_to_plot = 'L3Mi1'

#Plots settings

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 10}
axes = {'labelsize': 14, 'titlesize': 14}
ticks = {'labelsize': 10}
legend = {'fontsize': 12}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', **ticks)
plt.rc('ytick', **ticks)




#%% Initialyzing empty lists for common dataframe
all_columns_number_partners_dict_list = []
all_columns_length_dict_list = []
all_columns_norm_length_dict_list = []
all_columns_temp_final_input_output_dict_list = []

all_columns_centrality_df_list = []
all_columns_path_df_list = []
all_columns_final_input_df_list = []
all_columns_final_output_df_list = []
all_columns_final_input_ranked_df_list = []
all_columns_final_output_ranked_df_list = []
all_columns_weigth_line_df_list = []

#%% Loading and aggregating data from different files

i = len(pkl_file_list)  # Temp to denote one pickle file being uploaded
for files in range(i): #Looping across files
    pkl_data_path = os.path.join(pkl_folder,pkl_file_list[files])
    infile = open(pkl_data_path, 'rb') 
    data = pickle.load(infile)
    print('Loaded data for: %s' % data['user_parameters']['column'])

    #DataFrames
    temp_final_input_df = data['final_input_df']
    temp_final_output_df = data['final_output_df']
    temp_final_input_ranked_df = data['final_input_ranked_df']
    temp_final_output_ranked_df = data['final_output_ranked_df']
    temp_centrality_df = data['centrality_df']
    temp_path_df = data['path_df']
    temp_path_df['Path'] = temp_path_df['Path'].agg(lambda x: ','.join(map(str, x)))

    all_columns_final_input_df_list.append(temp_final_input_df)
    all_columns_final_output_df_list.append(temp_final_input_df)
    all_columns_final_input_ranked_df_list.append(temp_final_input_df)
    all_columns_final_output_ranked_df_list.append(temp_final_input_df)
    all_columns_centrality_df_list.append(temp_centrality_df)
    all_columns_path_df_list.append(temp_path_df)
    
    #Dictionaries
    temp_final_input_output_dict = data['final_input_output_dict']
    temp_number_partners_dict = data['number_partners_dict']
    temp_length_dict = data['length_dict']
    temp_norm_length_dict = data['norm_length_dict']

    all_columns_temp_final_input_output_dict_list.append(temp_final_input_output_dict)
    all_columns_number_partners_dict_list.append(temp_number_partners_dict)
    all_columns_length_dict_list.append(temp_length_dict)
    all_columns_norm_length_dict_list.append(temp_norm_length_dict)

    

# Combining dataframes
all_columns_centrality_df = pd.concat(all_columns_centrality_df_list)
all_columns_path_df = pd.concat(all_columns_path_df_list)
all_columns_final_input_df = pd.concat(all_columns_final_input_df_list)
all_columns_final_output_df = pd.concat(all_columns_final_output_df_list)
all_columns_final_input_ranked_df = pd.concat(all_columns_final_input_ranked_df_list)
all_columns_final_output_ranked_df = pd.concat(all_columns_final_output_ranked_df_list)

# all_columns_partners_df.reset_index(inplace = True)
# all_columns_weigth_df.reset_index(inplace = True)
# all_columns_partners_df.rename(columns = {'index':'Column index'}, inplace = True)
# all_columns_weigth_df.rename(columns = {'index':'Column index'}, inplace = True)

#%% Plotting
cm = 1/2.54  # centimeters in inches
_ci=68 # confidence interval of 68 is ~1 standard error

# Add at least 4 neurons, names must be written as in the data set!
list_of_neurons = ['Tm1','Tm2','Tm4', 'Tm9', 'Mi1','Tm3','Mi4', 'Mi9'] # 
list_of_neurons = ['Tm1','Tm2','Tm4', 'Tm9', 'Mi1'] # 
fig_heatmap, fig_heatmap_max, fig_heatmap_sum = heatmap_plot(short_col_names,all_columns_final_input_df,list_of_neurons,user_parameters,'Input')


############################ Bar plots with seaborn ##########################

variable_partners = '# of Partners'
variable_weigth = 'Weigth'

bar_fig2, bar_axes2 = plt.subplots(nrows= 3,ncols=2,figsize=(40*cm, 30*cm)) # All together

cur_df = all_columns_weigth_df.loc[all_columns_weigth_df['Connection']== 'Direct']
_color = 'blue'
_order=cur_df.sort_values(variable_weigth,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[0,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[0,0].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[0,0].spines['right'].set_visible(False)
bar_axes2[0,0].spines['top'].set_visible(False)            

cur_df = all_columns_partners_df.loc[all_columns_partners_df['Connection']== 'Direct']
_color = 'blue'
_order=cur_df.sort_values(variable_partners,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[0,1],x = 'Neuron', y = variable_partners, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[0,1].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[0,1].spines['right'].set_visible(False)
bar_axes2[0,1].spines['top'].set_visible(False)  

cur_df = all_columns_weigth_df.loc[all_columns_weigth_df['Connection']== 'Indirect 1']
_color = 'orange'
_order=cur_df.sort_values(variable_weigth,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[1,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[1,0].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[1,0].spines['right'].set_visible(False)
bar_axes2[1,0].spines['top'].set_visible(False)  

cur_df = all_columns_partners_df.loc[all_columns_partners_df['Connection']== 'Indirect 1']
_color = 'orange'
_order=cur_df.sort_values(variable_partners,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[1,1],x = 'Neuron', y = variable_partners, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[1,1].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[1,1].spines['right'].set_visible(False)
bar_axes2[1,1].spines['top'].set_visible(False)  

cur_df = all_columns_weigth_df.loc[all_columns_weigth_df['Connection']== 'Indirect 2']
_color = 'green'
_order=cur_df.sort_values(variable_weigth,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[2,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[2,0].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[2,0].spines['right'].set_visible(False)
bar_axes2[2,0].spines['top'].set_visible(False)  

cur_df = all_columns_partners_df.loc[all_columns_partners_df['Connection']== 'Indirect 2']
_color = 'green'
_order=cur_df.sort_values(variable_partners,ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes2[2,1],x = 'Neuron', y = variable_partners, data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes2[2,1].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes2[2,1].spines['right'].set_visible(False)
bar_axes2[2,1].spines['top'].set_visible(False) 

bar_axes2[0,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[0,1].set_ylabel('Partners', fontsize = 8)
bar_axes2[1,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[1,1].set_ylabel('Partners', fontsize = 8)
bar_axes2[2,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[2,1].set_ylabel('Partners', fontsize = 8)

_title = 'All columns ,' + graph + ': ' + ' - Direct- , - Indirect 1 - and -Indirect 2- connections'
bar_fig2.suptitle(_title, fontsize = 12)
plt.show()

############################ Centrality plots with seaborn #########################

#Plotting
fig_centrality = centrality_plot(all_columns_centrality_df,user_parameters)


bar_fig_c, bar_axes_c = plt.subplots(nrows= 2,ncols=2,figsize=(30*cm, 30*cm)) # All together

cur_df = all_columns_centrality_df
_color = 'tomato'
_order=cur_df.sort_values('Degree',ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes_c[0,0],x = 'Neuron', y = 'Degree', data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes_c[0,0].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes_c[0,0].spines['right'].set_visible(False)
bar_axes_c[0,0].spines['top'].set_visible(False) 

_color = 'purple'
_order=cur_df.sort_values('Pagerank eigenvector',ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes_c[0,1],x = 'Neuron', y = 'Pagerank eigenvector', data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes_c[0,1].set_xticklabels(_order, rotation = 90, fontsize = 7)
bar_axes_c[0,1].spines['right'].set_visible(False)
bar_axes_c[0,1].spines['top'].set_visible(False) 

_color = 'teal'
_order=cur_df.sort_values('Betweenness',ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes_c[1,0],x = 'Neuron', y = 'Betweenness', data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes_c[1,0].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes_c[1,0].spines['right'].set_visible(False)
bar_axes_c[1,0].spines['top'].set_visible(False) 

_color = 'gold'
_order=cur_df.sort_values('Closeness',ascending = False).Neuron
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes_c[1,1],x = 'Neuron', y = 'Closeness', data = cur_df, order=_order,
            color=_color, ci = _ci)
bar_axes_c[1,1].set_xticklabels(_order, rotation = 90, fontsize = _fontsize)
bar_axes_c[1,1].spines['right'].set_visible(False)
bar_axes_c[1,1].spines['top'].set_visible(False) 

_title = 'All columns ,' + graph + ': ' + ' centrality measures'
bar_fig_c.suptitle(_title, fontsize = 12)
plt.show()

############################ Pathplot with seaborn ##########################


bar_fig_p, bar_axes_p = plt.subplots(nrows= 1,ncols=1,figsize=(40*cm, 10*cm)) # All together

if node_to_node_to_plot[0:2] == 'L1' :
    _color = '#E64D00'
elif  node_to_node_to_plot[0:2] == 'L2' :
    _color = '#8019E8'
elif node_to_node_to_plot[0:2] == 'L3' :
    _color = '#1B9E77'

# cur_df = all_columns_path_df
cur_df = all_columns_path_df[all_columns_path_df['Node_to_node'] == node_to_node_to_plot]
_order=cur_df.sort_values('Weigth',ascending = False).Path
_order = _order.drop_duplicates(keep='first', inplace=False)
sns.barplot(ax = bar_axes_p,x = 'Path', y = 'Weigth', data = cur_df, order=_order,
            color=_color, ci = _ci)
plt.xticks(rotation=90)
_title = 'All columns ,' + graph + ': ' + ' paths ' + node_to_node_to_plot
bar_fig_p.suptitle(_title, fontsize = 12)

bar_axes_p.spines['right'].set_visible(False)
bar_axes_p.spines['top'].set_visible(False) 
plt.show()





#%% Saving data
if save_figures:
    bar_fig2.savefig(main_processed_data_folder+'\\All columns Bar plots.pdf',bbox_inches = "tight")
    bar_fig_c.savefig(main_processed_data_folder+'\\All columns Centrality measures.pdf',bbox_inches = "tight")
    bar_fig_p.savefig(main_processed_data_folder+'\\All columns paths %s.pdf' % node_to_node_to_plot,bbox_inches = "tight")

    print('Figures saved')