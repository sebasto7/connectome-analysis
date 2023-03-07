# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:16:36 2021

@author: Sebastian Molina-Obando

"""


#%% Importing packages
    
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from core_functions_network_analysis import path_length_transformation_plot, graph_plot,node_to_node_graph_analysis_and_plot
from core_functions_network_analysis import  input_output_plot,direct_indirect_connections_plot, connections_histomgram_plot,centrality_plot
from core_functions_network_analysis import Graph, centrality_analysis, direct_indirect_connections_analysis, input_output_analysis
from core_functions_general import saveWorkspace

#For Neuprint queries:
from neuprint import Client
from neuprint import fetch_adjacencies,  merge_neuron_properties
from neuprint import fetch_neurons, NeuronCriteria as NC

TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InNlYmFzdGlhbi5tb2xpbmEub2JhbmRvQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2hWZjMxR2RHeURzYmtfUW5qdW00b1U4SVZ5QTBEcXNVaXdNZ1ZrTEE9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MzA5NTQ5MTB9.YUJV-C5VoOZ-huLvc73EhWa6KWnejmemqzl9V-OrBKs'
c = Client('neuprint-examples.janelia.org', dataset='medulla7column', token= TOKEN)
c.fetch_version()


#%% Messages to developer
print('Tm3 ant and post instances only available in columns: C,D,E and home')

#%% User parameters in a dictionary
user_parameters = {}
user_parameters['dataPath']= r'D:\Connectomics-Data\datasets\NeuPrint\fib25'
user_parameters['file']='C_column_fib25.csv'#dataset file in cvs format. leave empty if calling data from NeuPrint
user_parameters['column']='-C' # before: 'F-column', 'home-column'; use better just: '-F', 'home'
user_parameters['graph']= 'Fib25_data_7medulla_columns'
user_parameters['aggregate']= False # For merging nodes with the same begginning of name of length = aggregate_ii
user_parameters['aggregate_ii']=4
user_parameters['start_node']= 'L3' # string of lenght = aggregate_ii
user_parameters['last_node']= 'Tm1'  # string of lenght = aggregate_ii
user_parameters['node_of_interest']= 'Tm1'  # for input-output plotting
user_parameters['_cutoff']=3 # max num of neurons between start and last node for path calculations, check "node_to_node_graph_analysis_and_plot"
user_parameters['neurons_to_exclude']= ['L1','L2','L3','T4a','T4b','T4c','T4d'] # exclude from centrality analysis and SSSP Analysis, not from the graph
user_parameters['multiple_start_nodes']= ['L1','L2','L3']
user_parameters['synaptic_stength_filter'] = 1 # connection number lower than this number will be discarded
user_parameters['defined_microcirtuit'] = ['Tm1','Tm2','Tm4','Tm9','CT1','C3', 'C2', 'T1'] # Used so far only for the stacked bar plot in "direct_indirect_connections_plot"



#%% Other user parameters:
load_from = 'NeuPrint' # 'NeuPrint' , 'CSV'
length_tranformation_function = 'linear_flip_function' # 'linear_flip_function', 'reciprocal_function' # Transformation for edge length between nodes (currently using synaptic count values) 
save_data = 1
save_figures = 1
plot_node_to_tode_paths = 0
main_data_folder = r'C:\Users\sebas\Documents\Connectomics-Data' # r'D:\Connectomics-Data'
split_columns = False # True only for data files where the info of all columns is mixed, code not finished

#%% Plot settings

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}
axes = {'labelsize': 16, 'titlesize': 16}
ticks = {'labelsize': 14}
legend = {'fontsize': 14}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', **ticks)
plt.rc('ytick', **ticks)

cm = 1/2.54  # centimeters in inches



#%% Loading the data of the circuit graph
if load_from == 'CSV':
    filePath = os.path.join(user_parameters['dataPath'], user_parameters['file'])
    Table = pd.read_csv(filePath)
    Table = Table[Table['N'] >user_parameters['synaptic_stength_filter']].copy()
    if split_columns:
        Table = Table[Table.PreSynapticNeuron.str.endswith(user_parameters['column'][0])]
        Table = Table[Table.PostSynapticNeuron.str.endswith(user_parameters['column'][0])]
elif load_from == 'NeuPrint':

    q = """\
        MATCH (n :Neuron)
        WHERE n.instance =~ '.*{}.*'
        RETURN n.bodyId AS bodyId, n.type as type, n.instance AS instance, n.pre AS numpre, n.post AS numpost
        ORDER BY n.pre + n.post DESC
    """
    column_id = user_parameters['column']
    results = c.fetch_custom(q.format(column_id))

    #print(results.head())

    # Fetchin connectivity among all nuerons in the data set
    neuron_ids = results['bodyId'].tolist()
    
    # All upstream and dowsntream connections of a set of neurons # AKA INPUTS AND OUTPUTS
    neuron_df, conn_df = fetch_adjacencies(neuron_ids, neuron_ids)
    Table = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
    Table.rename(columns = {'type_pre':'PreSynapticNeuron', 'type_post':'PostSynapticNeuron', 'weight':'N'}, inplace = True)
    Table = Table[Table['N'] >user_parameters['synaptic_stength_filter']].copy()
    #print(conn_df.head())

#Juan + Chatgpt code:
list_of_instances=['-C']
neuron_df=fetch_neurons(NC(inputRois=['distal','proximal']))[0] 
neuron_df = neuron_df.replace(to_replace='None', value=np.nan).dropna().copy()
#neuron_df = neuron_df[neuron_df['instance'].str.contains(user_parameters['column'])].copy()
neuron_df = neuron_df[neuron_df['instance'].str.contains('|'.join(list_of_instances))].copy()

#Quick sanity check for the presence of some neuron types
check = 'Tm3'
test_list =  Table['PostSynapticNeuron'].tolist() #neuron_df['instance'].unique().tolist()
res = [idx for idx in test_list if idx.startswith(check)]
#print(res)

#%% Auto creation of important paths
dirPath = os.path.join(main_data_folder, user_parameters['column'], user_parameters['graph'])
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
main_processed_data_folder = os.path.join(main_data_folder,'processed-data',user_parameters['graph'])


#%% Data visualization 
#Plotting path length transformations:
fig_path_lentgh_transform_reciprocal = path_length_transformation_plot(Table, user_parameters, 'reciprocal_function')
fig_path_lentgh_transform_flipped = path_length_transformation_plot(Table, user_parameters, 'linear_flip_function')

# Plotting histograms - Number of connections
fig_histograms = connections_histomgram_plot(Table, user_parameters)


#%% Graph creation
# Creating list of nodes
presynaptic_neurons = Table.PreSynapticNeuron.unique()
postsynaptic_neurons = Table.PostSynapticNeuron.unique()
neurons = list(np.unique(np.concatenate((presynaptic_neurons,postsynaptic_neurons), axis = 0)))
for i,n in enumerate(neurons):
    if n[-4:] == 'home':
        neurons[i] = n[0:-4]
#[n = n[0:-4] for n in neurons if n[-4:] == 'home'] # Why this list comprehension gives error?

neurons_list_aggregated = []
for neuron in neurons:
    temp_neuron = neuron[0:user_parameters['aggregate_ii']]
    if temp_neuron not in neurons_list_aggregated:
        neurons_list_aggregated.append(temp_neuron)
user_parameters['neurons_list_aggregated'] = neurons_list_aggregated

#Initialyzing and filling the graph with nodes and edges
customGraph = Graph()
if user_parameters['aggregate']:
    neuron_list = neurons_list_aggregated
else:
    neuron_list = neurons
user_parameters['neuron_list'] = neuron_list
    
for n,neuron in enumerate(neuron_list):
        customGraph.addNode(neuron)
        customGraph.addNode_num(n)
        
for index, row in Table.iterrows():
    # access data using column names
    if user_parameters['aggregate']:
        pair = row['PreSynapticNeuron'][0:user_parameters['aggregate_ii']],row['PostSynapticNeuron'][0:user_parameters['aggregate_ii']]
    else:
        if row['PreSynapticNeuron'][-4:] == 'home':
            _pre = row['PreSynapticNeuron'][0:-4]
            _post = row['PostSynapticNeuron'][0:-4]
        else:
            _pre = row['PreSynapticNeuron']
            _post = row['PostSynapticNeuron']

        pair = _pre ,_post 
        
    if pair in customGraph.distances:
        temp_N = customGraph.distances[pair] + row['N']
        # print('%f + %f = %f' % (customGraph.distances[pair],row['N'],temp_N))
        customGraph.addEdge(pair[0],pair[1],temp_N)
    else:
        customGraph.addEdge(pair[0],pair[1],row['N'])

#%% # Graph creation with networkkx
Weights = customGraph.distances
for key, value in Weights.items():
    Weights[key]= round(value) 

# Applying lenght transformation functions
if length_tranformation_function == 'reciprocal_function':
    edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()] 
elif length_tranformation_function == "linear_flip_function":
    edges = [(k[0], k[1], {'weight': (-v+int(max(Table['N']))+1)}) for k, v in Weights.items()] 

G = nx.DiGraph() 
#G = nx.Graph()
G.add_edges_from(edges)

aa = []
for i in range(1,len(edges)):
    aa.append(edges[i][2]['weight'])

# temporary untill fixing some code below
edges_plot = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()] 
G_plot = nx.DiGraph()
G_plot.add_edges_from(edges_plot)
    
#%% Visualizing the graph
fig_graph_original_length = graph_plot(Weights, user_parameters, 'none')
fig_graph_original_length_transformed = graph_plot(Weights, user_parameters, length_tranformation_function)


#%% Single source shortest path  (SSSP) analysis with NetwrokX

# >>> Relevant message: currently not interesting to print it out, moving to show it in a different way. Therefore is all commented out

# length, path = nx.single_source_dijkstra(G, user_parameters['start_node'])

# path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items())
# message_str = '\n >>>>>>> All paths: \n\n%s can reach a neuron via [shortest path]: \n\n%s ' % (user_parameters['start_node'], path_str)
# print(message_str)

# path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items() if k == user_parameters['last_node'])
# message_str = '\n >>>>>>> Single path: \n\n%s can reach %s via [shortest path]: \n\n%s ' % (user_parameters['start_node'],user_parameters['last_node'], path_str)
# print(message_str)


#%% Node to node analysis
#Highlights specific paths in the graph between two nodes and save a dataFrame with all the paths
path_df = node_to_node_graph_analysis_and_plot(G, Weights, user_parameters,dirPath,save_figures,plot_node_to_tode_paths=False)   

#TO DO: write analysis plot function for mean_path_length between pairs

#%% Centrality measures analysis
#Currently extracting: closeness_centrality, betweenness_centrality, degree_centrality, and eigenvector_centrality
centrality_df = centrality_analysis(G,user_parameters)
#Plotting
fig_centrality = centrality_plot(centrality_df,user_parameters)


#%% Direct and indirect connections analysis for all nodes
number_partners_dict, length_dict, norm_length_dict = direct_indirect_connections_analysis(Weights,user_parameters)
#Plotting
fig_direct_indirect_connections, fig_stacked_connections = direct_indirect_connections_plot(number_partners_dict,length_dict,norm_length_dict,user_parameters)

#%% Input- ouput analysis
final_input_output_dict, final_input_df,final_output_df, final_input_ranked_df ,final_output_ranked_df,final_input_ranked_norm_df,final_output_ranked_norm_df = input_output_analysis(Weights,user_parameters)

#Input - outputs plots for a node_of_interest
node_of_interest = user_parameters['node_of_interest']
input_output_fig, input_output_stacked_neuron_type_fig, input_output_stacked_ranked_fig, input_output_fractions_fig, input_output_line_fig = input_output_plot(node_of_interest,final_input_output_dict,final_input_df,final_output_df,final_input_ranked_df,final_output_ranked_df,final_input_ranked_norm_df,final_output_ranked_norm_df,user_parameters)



#%% Saving data
if save_figures:
        
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    save_dir = dirPath +'\\figures\\'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) # Seb: creating figures folder     
    fig_graph.savefig(save_dir+'Graph %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")        
    fig_stacked_connections.savefig(save_dir+'Stacked bar plot %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    fig_direct_indirect_connections.savefig(save_dir+'Bar plots %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    fig_centrality.savefig(save_dir+'Centrality measures %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    input_output_fig.savefig(save_dir+'Inputs and outputs - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    input_output_fractions_fig.savefig(save_dir+'Inputs and outputs - FRACTION  %s %s .pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    input_output_line_fig.savefig(save_dir+'Inputs and outputs FRACTION - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    print('\nFigures saved.\n')
    
if save_data:
    os.chdir(main_data_folder) # Seb: X_save_vars.txt file needs to be there    
    varDict = locals()
    pckl_save_name = ('%s' % user_parameters['column'])
    saveOutputDir = dirPath+ '\\processed-data\\' 
    if not os.path.exists(saveOutputDir): # Seb: creating processed data folder
        os.mkdir(saveOutputDir) 
    saveWorkspace(saveOutputDir,pckl_save_name, varDict,
                  varFile='main_connectome_analysis_save_vars.txt',extension='.pickle')
    if not os.path.exists(main_processed_data_folder):
        os.makedirs(main_processed_data_folder) # Seb: creating folder for storing all data per stim type
    saveWorkspace(main_processed_data_folder,pckl_save_name, varDict, 
                    varFile='main_connectome_analysis_save_vars.txt',extension='.pickle')
        
    print('\n\n%s processed-data saved.\n\n' % pckl_save_name)

    