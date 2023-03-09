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
from core_functions_network_analysis import  input_output_plot,direct_indirect_connections_plot, connections_histomgram_plot,centrality_plot,mean_path_length_plot
from core_functions_network_analysis import Graph, centrality_analysis, direct_indirect_connections_analysis, input_output_analysis
from core_functions_general import saveWorkspace

#For Neuprint queries:
from neuprint import Client
from neuprint import fetch_adjacencies,  merge_neuron_properties
from neuprint import fetch_neurons, NeuronCriteria as NC


def main_analysis(user_parameters):

    print('Main analysis starts:\n')

    #%% Neuprint settings
    TOKEN = user_parameters['NeuPrint_TOKEN']
    c = Client('neuprint-examples.janelia.org', dataset='medulla7column', token= TOKEN)
    c.fetch_version()


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
    split_columns = False # True only for data files where the info of all columns is mixed, code not finished
    if user_parameters['data_load_option'] == 'CSV':
        filePath = os.path.join(user_parameters['dataPath'], user_parameters['file'])
        Table = pd.read_csv(filePath)
        Table = Table[Table['N'] >user_parameters['synaptic_stength_filter']].copy()
        if split_columns:
            Table = Table[Table.PreSynapticNeuron.str.endswith(user_parameters['column'][0])]
            Table = Table[Table.PostSynapticNeuron.str.endswith(user_parameters['column'][0])]
    elif user_parameters['data_load_option'] == 'NeuPrint':

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

    #%% Some filtering
    Table = Table[(Table['PreSynapticNeuron'] != user_parameters['exclude_node']) & (Table['PostSynapticNeuron'] != user_parameters['exclude_node'])]
    #%% Auto creation of important paths
    dirPath = os.path.join(user_parameters['mainFolder'], user_parameters['column'], user_parameters['graph'])
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    main_processed_data_folder = os.path.join(user_parameters['mainFolder'],'processed-data',user_parameters['graph'])


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
    if user_parameters['edge_length_tranformation_function'] == 'reciprocal_function':
        edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()] 
    elif user_parameters['edge_length_tranformation_function'] == "linear_flip_function":
        edges = [(k[0], k[1], {'weight': (-v+int(max(Table['N']))+1)}) for k, v in Weights.items()] 

    G = nx.DiGraph() 
    #G = nx.Graph()
    G.add_edges_from(edges)

    # #Sanity check for above transformation
    # aa = []
    # for i in range(1,len(edges)):
    #     aa.append(edges[i][2]['weight'])
    # sns.histplot(x=aa, stat= 'counts', binwidth=1 )

    # temporary untill fixing some code below
    edges_plot = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()] 
    G_plot = nx.DiGraph()
    G_plot.add_edges_from(edges_plot)
        
    #%% Visualizing the graph
    fig_graph_original_length = graph_plot(Weights, user_parameters, 'none')
    fig_graph_original_length_transformed = graph_plot(Weights, user_parameters, user_parameters['edge_length_tranformation_function'])


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
    path_df = node_to_node_graph_analysis_and_plot(G, Weights, user_parameters,dirPath,user_parameters['save_figures'],user_parameters['plot_node_to_tode_paths'])   
    path_df['Norm_weigth'] = list(map(lambda x, y: y/(len(x)-1), path_df['Path'], path_df['Weigth'])) # Normalizing the weigth
    path_df['Jumps'] = list(map(lambda x: len(x)-1, path_df['Path'])) # Number of jumps between nodes

    #TO DO: write analysis plot function for mean_path_length between pairs
    fig_mean_path_lenght = mean_path_length_plot(path_df, user_parameters,user_parameters['save_figures'])

    agg_functions = ['sum', 'mean', 'median', 'min', 'max', 'std', 'sem','count', 'describe', 'size','first','last']
    mean_path_df = path_df.groupby(['Start_node', 'Last_node']).agg(agg_functions)

    fig,axs = plt.subplots(nrows = 2, ncols = 2, figsize=(30*cm, 30*cm))

    #First axis
    sns.histplot(x = path_df['Norm_weigth'], stat = 'probability', binwidth=1, ax=axs[1,0])
    axs[1,0].set_xlim([0,50])

    #sns.histplot(x = mean_path_df['Weigth']['mean']['Weigth'], stat = 'probability', binwidth=0.1, ax=axs[1,0])

    #Another axis
    sns.histplot(x = path_df['Jumps'], stat = 'probability', binwidth=1, ax=axs[1,1])
    #sns.histplot(x = mean_path_df['Jumps']['mean']['Jumps'], stat = 'probability', binwidth=0.1, ax=axs[1,1])
    fig.savefig(user_parameters['mainFolder']+'\Mi1_si.pdf')   




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
    if user_parameters['save_figures']:
            
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        save_dir = dirPath +'\\figures\\'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # Seb: creating figures folder     
        fig_graph_original_length_transformed.savefig(save_dir+'Graph %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")        
        fig_stacked_connections.savefig(save_dir+'Stacked bar plot %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        fig_direct_indirect_connections.savefig(save_dir+'Bar plots %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        fig_centrality.savefig(save_dir+'Centrality measures %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        input_output_fig.savefig(save_dir+'Inputs and outputs - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        input_output_fractions_fig.savefig(save_dir+'Inputs and outputs - FRACTION  %s %s .pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        input_output_line_fig.savefig(save_dir+'Inputs and outputs FRACTION - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
        print('\nFigures saved.\n')
        
    if user_parameters['save_data']:
        os.chdir(user_parameters['mainFolder']) # Seb: X_save_vars.txt file needs to be there    
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
    
    return print('Main analysis done.')

    