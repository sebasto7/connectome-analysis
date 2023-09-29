# -*- coding: utf-8 -*-
"""
Created on Wed March 8 (Women's day! :)) 11:48:00 2023

@author: Sebastian Molina-Obando

"""

#%% Importing packages
from main_connectome_analysis import main_analysis

#%% Messages to developer
print('\nTemporary messages:')
print('Tm3 ant and post instances only available in columns: C,D,E and home\n')

#%% User parameters in a dictionary
user_parameters = {}
PC_disc = 'D'

user_parameters['data_load_option'] = 'NeuPrint' # 'NeuPrint' , 'CSV'
user_parameters['mainFolder']= f'{PC_disc}:\Connectomics-Data\FIB25'

user_parameters['column']='-F' # before: 'F-column', 'home-column'; use better just: '-F', 'home'
user_parameters['graph']= 'FIB25-Neuprint' # 'FIB25'
user_parameters['add_neurons'] = True
user_parameters['neurons_prefixes'] = ['T4a', 'T4b', 'T4c', 'T4d']
user_parameters['neurons_sufix'] = ['home']

# For data option 'CSV'
user_parameters['dataPath']= f'{PC_disc}:/Connectomics-Data/FIB25/datasets/NeuPrint/fib25'
user_parameters['file']='home_column_fib25.csv'#dataset file in cvs format. leave empty if calling data from NeuPrint

# Analysis options
user_parameters['edge_length_tranformation_function'] = 'reciprocal_function' # 'linear_flip_function', 'reciprocal_function' #'binary' #Transformation for edge length between nodes (currently using synaptic count values)
user_parameters['save_data'] = 1
user_parameters['save_figures'] = 1
user_parameters['plot_node_to_tode_paths'] = 0
user_parameters['aggregate']= False # For merging nodes with the same begginning of name of length = aggregate_ii
user_parameters['aggregate_ii']=4
user_parameters['start_node']= 'L1' # string of lenght = aggregate_ii
user_parameters['last_node']= 'T4a'  # string of lenght = aggregate_ii
user_parameters['node_of_interest']= 'Mi1'  # for input-output plotting
user_parameters['_cutoff']=3 # max num of neurons between start and last node for path calculations, check "node_to_node_graph_analysis_and_plot"
user_parameters['neurons_to_exclude']= [''] #['L1','L2','L3','T4a','T4b','T4c','T4d'] # exclude from centrality analysis and SSSP Analysis, not from the graph
user_parameters['multiple_start_nodes']= ['L1','L2','L3']
user_parameters['exclude_node']='' # 'Mi1'
user_parameters['synaptic_stength_filter'] = 1 # connection number lower than this number will be discarded
user_parameters['defined_microcirtuit'] = ['L1','L3','L5','Mi1','Tm3','CT1','C3', 'C2', 'Mi4', 'Mi9','T4a','T4b','T4c','T4d'] # Used so far only for the stacked bar plot in "direct_indirect_connections_plot"
#user_parameters['defined_microcirtuit'] = ['L1','L3','L5','Mi1','Tm3','CT1','C3', 'C2', 'Mi4', 'Mi9'] # Used so far only for the stacked bar plot in "direct_indirect_connections_plot"

# TOKEN
user_parameters['NeuPrint_TOKEN']= 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InNlYmFzdGlhbi5tb2xpbmEub2JhbmRvQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2hWZjMxR2RHeURzYmtfUW5qdW00b1U4SVZ5QTBEcXNVaXdNZ1ZrTEE9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MzA5NTQ5MTB9.YUJV-C5VoOZ-huLvc73EhWa6KWnejmemqzl9V-OrBKs'


#%% Running the analysis 
main_analysis(user_parameters)