# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 08:50:02 2021

@author: Sebastian Molina-Obando
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import PercentFormatter
import os

# General variables
cm = 1/2.54  # centimeters in inches

#Initializing the Graph Class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.nodes_num = []
        self.edges = defaultdict(list)
        self.distances = {}

    
    def addNode(self,value):
        self.nodes.add(value)
    def addNode_num(self, value):
        self.nodes_num.append(value)
    
    def addEdge(self, fromNode, toNode, distance):
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance
        

        

#Implementing Dijkstra's Algorithm

#Single Source Shortest Path (SSSP) - Dijkstra's algorithm
#code source: https://pythonwife.com/dijkstras-algorithm-in-python/

def dijkstra(graph, initial):
    visited = {initial : 0}
    path = defaultdict(list)

    nodes = set(graph.nodes)

    while nodes:
        minNode = None
        for node in nodes:
            if node in visited:
                if minNode is None:
                    minNode = node
                elif visited[node] < visited[minNode]:
                    minNode = node
        if minNode is None:
            break

        nodes.remove(minNode)
        currentWeight = visited[minNode]

        for edge in graph.edges[minNode]:
            weight = currentWeight + graph.distances[(minNode, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge].append(minNode)
    
    return visited, path

#%% Plotting function

def path_length_transformation_plot(Table, user_parameters, transformation_function):
    '''
    Transformation functions:
    - reciprocal_function: The reciprocal function: y = 1/x. For every x except 0, y represents its multiplicative inverse.
    - none

    '''
    # Line plot path length transformation

    x = list(range(1,len(Table.N)+1))
    y = list(Table.N) # Connections
    if transformation_function == "reciprocal_function":
        y2 = [1/i for i in y]  # 1/x function -> new distances
        x2 = [1/i for i in x]  #  1/x function
        y_des = sorted(y,reverse=True)
        y_as = sorted(y,reverse=False)
        y2_des = sorted(y2,reverse=True)
        y2_as = sorted(y2,reverse=False)
        x2_as = sorted(x2,reverse=False)

        d = {} # Getting frequency of connections
        for i in y:
            d[i] = d.get(i,0) + 1
        import operator
        d_des = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
        y_freq = list(d_des.values())
        y_freq_norm = [float(i)/max(y_freq) for i in y_freq]
        x_rank = range(1,len(y_freq)+1)

        #Linear regression
        m, c = np.polyfit(np.log10(x_rank), np.log10(y_freq), 1, w=np.sqrt(y_freq)) # fit log(y) = m*log(x) + c
        y_freq_fit = np.exp(m*np.log10(x_rank)+ c) # calculate the fitted values of y 

        fig, axes = plt.subplots(nrows= 2,ncols=2,figsize=(30*cm, 30*cm))

        axes[0,0].plot(y_as,y2_des) # Edge distance transformation
        axes[0,0].set_ylabel('Edge (distance)', fontsize = 12)
        axes[0,0].set_xlabel('Number of connections', fontsize = 12)

        axes[1,0].plot(y_as,y2_des) # Power law
        axes[1,0].set_ylabel('Edge (distance)', fontsize = 12)
        axes[1,0].set_yscale('log')
        axes[1,0].set_xscale('log')
        axes[1,0].set_xlabel('Number of connections', fontsize = 12)

        axes[0,1].scatter(x_rank,y_freq,s=30, alpha=0.7, edgecolors="k") # Zipf law connections
        axes[0,1].set_xlabel('Connection rank', fontsize = 12)
        axes[0,1].set_ylabel('Frequency', fontsize = 12)

        axes[1,1].scatter(x_rank,y_freq,s=30, alpha=0.7, edgecolors="k") # Zipf law connections
        #axes[2].plot(x_rank, y_freq, color = 'r')
        #axes[2].plot(x_rank, y_freq_fit, ':')
        axes[1,1].set_xlabel('Connection rank', fontsize = 12)
        axes[1,1].set_ylabel('Frequency', fontsize = 12)
        axes[1,1].set_yscale('log')
        axes[1,1].set_xscale('log')

        axes[0,0].spines['right'].set_visible(False)
        axes[0,0].spines['top'].set_visible(False)
        axes[1,0].spines['right'].set_visible(False)
        axes[1,0].spines['top'].set_visible(False)
        axes[0,1].spines['right'].set_visible(False)
        axes[0,1].spines['top'].set_visible(False)
        axes[1,1].spines['right'].set_visible(False)
        axes[1,1].spines['top'].set_visible(False)


        _title = f"Edge definition, {user_parameters['graph']} : {user_parameters['column']}"
        fig.suptitle(_title, fontsize = 12)

    print("Line plots for path length transformation done.")
    return fig


def connections_histomgram_plot(Table,user_parameters):
    

    fig, axes = plt.subplots(nrows= 2,ncols=1,figsize=(30*cm, 30*cm))
    axes[0].hist( list(Table.N), bins= 50, density=True)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Frequency', fontsize = 12)
    axes[0].set_xlabel('Number of connections', fontsize = 12) 


    #list_norm = [float(i)/max(list(Table.N)) for i in list(Table.N)]
    #axes[1].hist( list_norm, bins= 50, cumulative=True, density = True)
    axes[1].hist( list(Table.N), bins= 50, cumulative=True, density = True)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel('Cumulative', fontsize = 12)
    axes[1].set_xlabel('Number of connections', fontsize = 12) 
    #axes[1].set_xlabel('Percentatge of connections', fontsize = 12)
    #axes[1].xaxis.set_major_formatter(PercentFormatter(1))

    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    _title = f"Number of connections, {user_parameters['graph']} : {user_parameters['column']}"
    fig.suptitle(_title, fontsize = 12)

    return fig

def graph_plot(Weights, user_parameters, transformation_function):
    '''
    # documentation: https://networkx.org/documentation/stable/reference/introduction.html
    # examples: https://algorithmx-python.readthedocs.io/en/latest/examples/networkx-examples.html

    Transformation functions:
    - reciprocal_function: The reciprocal function: y = 1/x. For every x except 0, y represents its multiplicative inverse.
    - none

    '''

    concentricGraph = False # To generate a graph based on layers

    if transformation_function == "reciprocal_function":
        edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()] # Applied transformation
    elif transformation_function == "none":
        edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()]

    G = nx.DiGraph()
    # each edge is a tuple of the form (node1, node2, {'weight': weight})
    G.add_edges_from(edges)

    # Dividing all distance by 1 to define the strongest connection as the shortest distance.
    #customGraph.distances = {k:1/v for k, v in customGraph.distances.items()}

    # Choosing layout for the nodes (positions)
    retina_shell = []
    lamina_shell =[]
    medulla_shell = []
    lobula_plate_shell = []
    rest_shell = []
    if concentricGraph:
        for node in user_parameters['neurons_list_aggregated']:
            if node[0:1] == 'R':
                retina_shell.append(node)
            elif node[0:1] == 'L':
                lamina_shell.append(node)
            elif node[0:2] == 'Mi' or node[0:2] == 'Tm' or node[0:1] == 'C' or node[0:2] == 'T1' or node[0:2] == 'T2':
                medulla_shell.append(node)
            elif node[0:2] == 'T4':
                lobula_plate_shell.append(node)
            else:
                rest_shell.append(node)
                
        #shells = [retina_shell, lamina_shell,medulla_shell,lobula_plate_shell,rest_shell]
        shells = [lobula_plate_shell,medulla_shell,lamina_shell,retina_shell]
        pos = nx.shell_layout(G, shells)
    else:
        #pos = nx.spring_layout(G) # positions for all nodes
        # pos = nx.spring_layout(G, pos = initial_node_pos, fixed = initial_node_pos ) # positions for all nodes
        #pos = nx.random_layout(G)
        pos = nx.circular_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G) 
        #pos = nx.spiral_layout(G)


    fig,axes= plt.subplots(figsize=(20*cm, 20*cm))
    fig.suptitle(user_parameters['file'])

    ## nodes
    nx.draw_networkx_nodes(G,pos,node_size=600)

    ## labels
    nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')

    ## edges
    #nx.draw_networkx_edges(G,pos,edgelist=edges, width=2,connectionstyle="arc3,rad=-0.2")
    nx.draw_networkx_edges(G,pos,edgelist=edges, width=2)

    ## weights
    #labels = nx.get_edge_attributes(G,'weight')
    # for key, value in labels .items():
    #     labels [key]= round(1/value)
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,label_pos = 0.3,font_size=10)

    return fig

def node_to_node_graph_analysis_and_plot(G, Weights, user_parameters,dirPath,save_figures,plot_node_to_tode_paths):
    '''
    
    '''

    message_str = '\n >>>>>>> All paths in a distance of max %d neurons' % (user_parameters['_cutoff'])
    print(message_str)
    multiple_path_dict = {} #For multiple start nodes to last node
    for node in user_parameters['multiple_start_nodes']:
        start_node = node
        path_list = [] # For single start node to last node
        for path in nx.all_simple_paths(G, source=start_node , target=user_parameters['last_node'], cutoff=user_parameters['_cutoff']):
            path_list.append(path)
            print(path)
        _key = node+'_'+user_parameters['last_node']
        multiple_path_dict[_key] = path_list

    path_df = pd.DataFrame(columns =['Node_to_node' ,'Path', 'Weigth'])
    for key, value in multiple_path_dict.items():
        node_to_lode_list = [key]*len(value)
        path_list = value
        path_weigth_list = []
        for cur_path in path_list:
            
            
            # Get position using spring layout
            pos = nx.circular_layout(G)
            
            # Get shortest path
            highlighted_path = nx.shortest_path(G,source=user_parameters['start_node'],target=user_parameters['last_node'])
            highlighted_path = cur_path
            
            temp_weigth  = 0
            for i in range(len(highlighted_path)-1):
                temp_weigth = temp_weigth +(Weights[(highlighted_path[i],highlighted_path[i+1])])
            weigth_path = round(temp_weigth/(len(highlighted_path)-1))
            path_weigth_list.append(weigth_path)
            
            
            if plot_node_to_tode_paths:
                
                path_fig,axes= plt.subplots(figsize=(20*cm, 20*cm))
                
                if highlighted_path[0] == 'L1-' :
                    _color = '#E64D00'
                elif  highlighted_path[0] == 'L2-' :
                    _color = '#8019E8'
                elif highlighted_path[0] == 'L3-' :
                    _color = '#1B9E77'
                else:
                    _color = 'r'
                    
                path_edges = list(zip(highlighted_path,highlighted_path[1:]))
                
                # Draw nodes and edges not included in path
                nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes)-set(highlighted_path),node_size=600)
                #nx.draw_networkx_edges(G, pos, edgelist=set(G.edges)-set(path_edges), connectionstyle='arc3, rad = 0.3')
                nx.draw_networkx_edges(G, pos, edgelist=set(G.edges)-set(path_edges),width=2)
                
                # Draw nodes and edges included in path
                nx.draw_networkx_nodes(G, pos, nodelist=highlighted_path, node_color=_color,node_size=900)
                #nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r', connectionstyle='arc3, rad = 0.3')
                nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color=_color,width=8)
                
                # Draw labels
                path_fig.suptitle('Node-to-node path. Averaged weigth: %.1f  %s ' % (weigth_path,user_parameters['graph']))
                nx.draw_networkx_labels(G,pos)
                plt.axis('equal')
                plt.show()
                plt.close()
            
                if save_figures:
                    name_nodes = user_parameters['start_node'] +' to ' + user_parameters['last_node']
                    save_dir = dirPath +'\\figures\\paths\\' + name_nodes + '\\'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir) # Seb: creating figures folder
                    if len(highlighted_path) == 3:
                        path_fig.savefig(save_dir+'weigth %.1f path %s-%s-%s %s %s.pdf' % (weigth_path,user_parameters['start_node'],highlighted_path[1],user_parameters['last_node'],user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")       
                    elif len(highlighted_path) == 4:
                        path_fig.savefig(save_dir+'weigth %.1f path %s-%s-%s-%s %s %s.pdf' % (weigth_path,user_parameters['start_node'],highlighted_path[1],highlighted_path[2],user_parameters['last_node'],user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
                    
            # Creating paths dataframe
            cur_df = pd.DataFrame(list(zip(node_to_lode_list,path_list, path_weigth_list)),
                    columns =['Node_to_node' ,'Path', 'Weigth'])
            # Concatenating dataframes    
            path_df = path_df.append(cur_df)
        print('Node to node path analysis done.')
        return path_df

def centrality_analysis(G,user_parameters):
    '''
    
    '''
    from networkx.algorithms.centrality import closeness_centrality
    from networkx.algorithms.centrality import betweenness_centrality
    from networkx.algorithms.centrality import degree_centrality
    from networkx.algorithms.centrality import eigenvector_centrality

    # with NetwrokX
    degree_dict = degree_centrality(G)
    eigenvector_dict = eigenvector_centrality(G)
    pagerank_dict = nx.pagerank(G, alpha = 0.8)
    betweenness_dict = betweenness_centrality(G, k=None, normalized=True, weight="weight") # Use distances as weight?
    closeness_dict = closeness_centrality(G, u=None, distance="weight", wf_improved=True) # Inputs distances?

    degree_df =pd.DataFrame(degree_dict, index=[0]) 
    eigenvector_df =pd.DataFrame(eigenvector_dict, index=[0])
    pagerank_df =pd.DataFrame(pagerank_dict, index=[0])  
    betweenness_df =pd.DataFrame(betweenness_dict, index=[0]) 
    closeness_df =pd.DataFrame(closeness_dict, index=[0]) 

    degree_df = degree_df.T
    eigenvector_df = eigenvector_df.T
    pagerank_df = pagerank_df.T
    betweenness_df = betweenness_df.T
    closeness_df = closeness_df.T

    centrality_df = pd.concat([degree_df , eigenvector_df, pagerank_df,betweenness_df,closeness_df ], axis=1)
    centrality_df.columns = ['Degree', 'Eigenvector', 'Pagerank eigenvector', 'Betweenness', 'Closeness']
    centrality_df.reset_index(level=0, inplace=True)
    centrality_df.rename(columns = {'index':'Neuron'}, inplace = True)

    centrality_df = centrality_df[centrality_df.Neuron.isin(user_parameters['neurons_to_exclude']) == False]

    print('Centrality measures analysis done.')
    return centrality_df

def direct_indirect_connections_analysis(Weights,user_parameters):
    '''
    Direct and indirect connections analysis for all nodes.
    Edges considered to as the number of connections (x) and not the reciprocal (1/x)

    '''

    #number_of_all_partners = len(path) -1
    interconnections = 3 # Direct + indirect connections to asses (e.g., "3" gives the direct conections + and indirect with 1 and 2 neurons in between)
    number_partners_dict = {} # Number of synpatic partners for each type of interconncetion
    length_dict = {} # Total length of connections
    norm_length_dict = {} # Normalized length to number of contact partners

    edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()] 
    G = nx.DiGraph()
    G.add_edges_from(edges)

    for neuron in user_parameters['neuron_list']:
        if neuron in user_parameters['neurons_to_exclude']:
            continue
        length, path = nx.single_source_dijkstra(G, neuron)
        #length_2, path_2 = nx.single_source_dijkstra(G_plot, neuron)
        temp_connections_list = []
        temp_total_length_list = []
        temp_total_norm_length_list = []
        for i in range(interconnections):
            temp_total_length = 0
            temp_number_of_partners = 0
            for key in path.keys():
                if len(path[key])==2+i:
                    temp_number_of_partners += 1
                    temp_total_length = temp_total_length + length[key] # was  length_2[key]
            temp_connections_list.append(temp_number_of_partners)
            temp_total_length_list.append(temp_total_length)
            if temp_number_of_partners == 0:
                temp_total_norm_length_list.append(None)
            else:
                temp_total_norm_length_list.append(temp_total_length/temp_number_of_partners)
                
        number_partners_dict[neuron] = temp_connections_list
        length_dict[neuron] = temp_total_length_list
        norm_length_dict[neuron] = temp_total_norm_length_list

    print('Direct and indirect connections analysis done.')
    return number_partners_dict, length_dict, norm_length_dict
    
def input_output_analysis(Weights,user_parameters):
    '''
    Input- ouput analysis
    Edges considered to as the number of connections (x) and not the reciprocal (1/x)

    '''

    edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()] 

    final_input_output_dict = {}
    final_input_df = pd.DataFrame()  
    final_output_df = pd.DataFrame() 
    final_input_ranked_df = pd.DataFrame() # In ranked dfs, neuron names are replace with numbers deptcting first, second, etc.., inputs based on #of synaptic contacts
    final_output_ranked_df = pd.DataFrame() 
    for neuron in user_parameters['neuron_list']:
        #neuron_of_interest = user_parameters['node_of_interest']
        neuron_of_interest = neuron
        input_neurons_dict = {}
        output_neurons_dict = {}
        temp_input_output_dict = {}

        for item in edges:
            input_neuron = item[0]
            output_neuron = item[1]
            _weigth = float(item[2]['weight'])
            if input_neuron == neuron_of_interest:
                output_neurons_dict[output_neuron]=_weigth
            if output_neuron == neuron_of_interest:
                input_neurons_dict[input_neuron]=_weigth
        # Ordering the dictionaries based on value (here number of synaptic connections)        
        input_neurons_list = sorted(input_neurons_dict.items(), key=lambda x: x[1], reverse=True)
        output_neurons_list= sorted(output_neurons_dict.items(), key=lambda x: x[1], reverse=True)
        
        input_neurons_df = pd.DataFrame(input_neurons_dict, index=[0])
        input_neurons_df  = input_neurons_df .T
        input_neurons_df.rename(columns = {0:'Inputs'}, inplace = True)
        input_neurons_df.reset_index(level=0, inplace=True)
        input_neurons_df.rename(columns = {'index':'Neuron'}, inplace = True)
        output_neurons_df = pd.DataFrame(output_neurons_dict, index=[0])
        output_neurons_df  = output_neurons_df .T
        output_neurons_df.rename(columns = {0:'Outputs'}, inplace = True)
        output_neurons_df.reset_index(level=0, inplace=True)
        output_neurons_df.rename(columns = {'index':'Neuron'}, inplace = True)
        
        temp_input_output_dict['input_neurons'] = input_neurons_df
        temp_input_output_dict['output_neurons'] = output_neurons_df
        final_input_output_dict[neuron] = temp_input_output_dict
        
        
        # Input and output dataframes for all neurons
        temp_input_neurons_df = pd.DataFrame(input_neurons_dict, index=[0])
        temp_input_neurons_df = temp_input_neurons_df.rename(index={0:neuron})
        final_input_df=final_input_df.append(temp_input_neurons_df)
        
        temp_output_neurons_df = pd.DataFrame(output_neurons_dict, index=[0])
        temp_output_neurons_df = temp_output_neurons_df.rename(index={0:neuron})
        final_output_df=final_output_df.append(temp_output_neurons_df)
        
        # Input and output dataframes ranked 
        temp_input_ranked_df = temp_input_neurons_df.sort_values(by = neuron, axis=1, ascending=False)
        for i, column in enumerate(temp_input_ranked_df):
            temp_input_ranked_df.rename(columns = {column:i+1}, inplace = True)
        final_input_ranked_df=final_input_ranked_df.append(temp_input_ranked_df)
        
        temp_output_ranked_df = temp_output_neurons_df.sort_values(by = neuron, axis=1, ascending=False)
        for i, column in enumerate(temp_output_ranked_df):
            temp_output_ranked_df.rename(columns = {column:i+1}, inplace = True)
        final_output_ranked_df=final_output_ranked_df.append(temp_output_ranked_df)
        
    # Calculating proportions / normalizing data to the sum of all inputs or outputs
    final_input_ranked_norm_df = final_input_ranked_df.copy()  
    sum_df = final_input_ranked_norm_df.sum(axis=1)
    for neuron in final_input_ranked_norm_df.index:
        for column in final_input_ranked_norm_df.loc[[neuron]]:
            # Replacing the value of a given row (neuron) and column per its fraction
            final_input_ranked_norm_df.at[neuron, column] = (final_input_ranked_norm_df.loc[neuron, column]/sum_df[neuron])
            
    final_output_ranked_norm_df = final_output_ranked_df.copy()  
    sum_df = final_output_ranked_norm_df.sum(axis=1)
    for neuron in final_output_ranked_norm_df.index:
        for column in final_output_ranked_norm_df.loc[[neuron]]:
            # Replacing the value of a given row (neuron) and column per its fraction
            final_output_ranked_norm_df.at[neuron, column] = (final_output_ranked_norm_df.loc[neuron, column]/sum_df[neuron])
    
    print('Input-Ouput analysis done.')
    return final_input_output_dict, final_input_df, final_output_df, final_input_ranked_df, final_output_ranked_df,final_input_ranked_norm_df,final_output_ranked_norm_df
            

def autolabel(ax, variable, ascending = True):
        '''
        
        '''
        rects = ax.patches
        if not ascending:
            variable = variable.sort_values(ascending = False)
            variable = variable.to_list()
        for i, rect in enumerate(rects):
            height = rect.get_height()
            percent = f'{variable[i]} %'
            ax.text(rect.get_x() + rect.get_width() / 2., 0.2 * height,
                    percent,
                    ha='center', va='bottom', rotation=90, color='k')
    
def input_output_plot(node_of_interest,final_input_output_dict,final_input_df,final_output_df,final_input_ranked_df,final_output_ranked_df,final_input_ranked_norm_df,final_output_ranked_norm_df,user_parameters):

    '''
    
    '''
    
    ####################### Bar plots with seaborn  ######################
    # Input and output absolute number of connections (bar) and in percentatge as text

    #Getting the data for the node_of_interest
    input_df = final_input_output_dict[node_of_interest]['input_neurons']
    output_df = final_input_output_dict[node_of_interest]['output_neurons']

    input_percentatge = (input_df['Inputs']/sum(input_df['Inputs']))*100
    input_df['Percentage'] = input_percentatge
    input_df['Percentage']= input_df['Percentage'].round(2)
    output_percentatge = (output_df['Outputs']/sum(output_df['Outputs']))*100
    output_df['Percentage'] = output_percentatge
    output_df['Percentage']= output_df['Percentage'].round(2)

    #Bar plots with seaborn
    fig, axes = plt.subplots(nrows= 2,ncols=1,figsize=(30*cm, 20*cm)) # All together
    cur_df = input_df
    _color = 'blue'
    sns.barplot(ax = axes[0],x = 'Neuron', y = 'Inputs', data = cur_df, order=cur_df.sort_values('Inputs',ascending = False).Neuron,
                color=_color)

    axes[0].set_xticklabels(cur_df.sort_values('Inputs',ascending = False).Neuron.tolist(), rotation = 90, fontsize = 7)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    autolabel(axes[0],cur_df['Percentage'],ascending = False)


    cur_df = output_df
    _color = 'green'
    sns.barplot(ax = axes[1],x = 'Neuron', y = 'Outputs', data = cur_df, order=cur_df.sort_values('Outputs',ascending = False).Neuron,
                color=_color)
    axes[1].set_xticklabels(cur_df.sort_values('Outputs',ascending = False).Neuron.tolist(), rotation = 90, fontsize = 7)       
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    autolabel(axes[1],cur_df['Percentage'],ascending = False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' - '+ user_parameters['node_of_interest']
    fig.suptitle(_title, fontsize = 12)


    #################### Stacked bar plot with pandas  ###################
    # Inputs and outputs synapses per neuron type
    _fontsize = 6
    fig_s, axes_s = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

    weigth_df_bar = final_input_df
    #weigth_df_bar= weigth_df_bar.fillna(0)
    weigth_df_bar.plot(ax=axes_s[0],kind='bar', stacked=True, fontsize = _fontsize)
    weigth_df_bar = final_output_df
    #weigth_df_bar= weigth_df_bar.fillna(0)
    weigth_df_bar.plot(ax=axes_s[1],kind='bar', stacked=True, fontsize = _fontsize)


    axes_s[0].set_ylabel('Input synapses per neuron type', fontsize = 12)
    axes_s[1].set_ylabel('Output synapses per neuron type', fontsize = 12)

    axes_s[0].spines['right'].set_visible(False)
    axes_s[1].spines['right'].set_visible(False)
    axes_s[0].spines['top'].set_visible(False)
    axes_s[1].spines['top'].set_visible(False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by synaptic partner'
    fig_s.suptitle(_title, fontsize = 12)


    #################### Stacked bar plot with pandas  ###################
    #Inputs and outputs synapses per neuron, RANKED, first "N" connections
    
    N = 20
    # Sixth barplot
    _fontsize = 6
    fig_r, axes_r = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

    weigth_df_bar = final_input_ranked_df.iloc[: , :N]
    weigth_df_bar.plot(ax=axes_r[0],kind='bar', stacked=True, fontsize = _fontsize)
    weigth_df_bar = final_output_ranked_df.iloc[: , :N]
    weigth_df_bar.plot(ax=axes_r[1],kind='bar', stacked=True, fontsize = _fontsize)


    axes_r[0].set_ylabel('Input synapses, ranked', fontsize = 12)
    axes_r[1].set_ylabel('Output synapses, ranked', fontsize = 12)

    axes_r[0].spines['right'].set_visible(False)
    axes_r[1].spines['right'].set_visible(False)
    axes_r[0].spines['top'].set_visible(False)
    axes_r[1].spines['top'].set_visible(False)

    axes_r[0].get_legend().remove()
    axes_r[1].get_legend().remove()

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by connection number'
    fig_r.suptitle(_title, fontsize = 12)


    #################### Stacked bar plot with pandas  ###################
    #Normalized to the sum of input. Input-output FRACTIONS

    # Seventh barplot
    fig_f, axes_f = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together
    _fontsize = 6

    weigth_df_bar = final_input_ranked_norm_df.iloc[: , :N]
    weigth_df_bar.plot(ax=axes_f[0],kind='bar', stacked=True, fontsize = _fontsize)
    weigth_df_bar = final_output_ranked_norm_df.iloc[: , :N]
    weigth_df_bar.plot(ax=axes_f[1],kind='bar', stacked=True, fontsize = _fontsize)


    axes_f[0].set_ylabel('Fraction of input synapses', fontsize = 12)
    axes_f[1].set_ylabel('Fraction of output synapses', fontsize = 12)

    axes_f[0].spines['right'].set_visible(False)
    axes_f[1].spines['right'].set_visible(False)
    axes_f[0].spines['top'].set_visible(False)
    axes_f[1].spines['top'].set_visible(False)

    axes_f[0].get_legend().remove()
    axes_f[1].get_legend().remove()

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' FRACTIONS'
    fig_f.suptitle(_title, fontsize = 12)


    ############################ Line plots with pandas######################
    # First "N" connections in absolute numbers
    # TODO: add relative quantification as well and measure slopes fitting and exponential curve

    N = 7

    #Firt line plot
    fig_l, axes_l = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together
    _fontsize = 6
    weigth_line_df= final_input_ranked_norm_df.fillna(0)
    weigth_line_df=weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
    weigth_line_df.T.plot(ax=axes_l[0])

    weigth_line_df= final_output_ranked_norm_df.fillna(0)
    weigth_line_df=weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
    weigth_line_df.T.plot(ax=axes_l[1])

    axes_l[0].set_ylabel('Fraction of input synapses', fontsize = _fontsize)
    axes_l[1].set_ylabel('Fraction of output synapses', fontsize = _fontsize)

    axes_l[0].spines['right'].set_visible(False)
    axes_l[1].spines['right'].set_visible(False)
    axes_l[0].spines['top'].set_visible(False)
    axes_l[1].spines['top'].set_visible(False)
    axes_l[0].set_ylim(0,1)
    axes_l[1].set_ylim(0,1)


    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ': ' + user_parameters['node_of_interest'] +' FRACTIONS'
    fig_l.suptitle(_title, fontsize = 12)

    return fig, fig_s, fig_r, fig_f, fig_l
        
def direct_indirect_connections_plot(number_partners_dict,length_dict,norm_length_dict,user_parameters):
    '''
    
    '''
    #Pandas dataframes creation
    variable_partners = '# of Partners'
    partners_df = pd.DataFrame(number_partners_dict)
    partners_df  = partners_df .T
    partners_df.columns =['Direct', 'Indirect 1', 'Indirect 2']
    partners_df= partners_df.stack().reset_index()
    partners_df.rename(columns = {'level_0':'Neuron'}, inplace = True)
    partners_df.rename(columns = {'level_1':'Connection'}, inplace = True)
    partners_df.rename(columns = {0:variable_partners}, inplace = True)

    variable_weigth = 'Weigth'
    weigth_df = pd.DataFrame(length_dict)
    weigth_df  = weigth_df .T
    weigth_df.columns =['Direct', 'Indirect 1', 'Indirect 2']
    weigth_df= weigth_df.stack().reset_index()
    weigth_df.rename(columns = {'level_0':'Neuron'}, inplace = True)
    weigth_df.rename(columns = {'level_1':'Connection'}, inplace = True)
    weigth_df.rename(columns = {0:variable_weigth}, inplace = True)

    variable = 'Norm weigth'
    df = pd.DataFrame(norm_length_dict)
    df  = df .T
    df.columns =['Direct', 'Indirect 1', 'Indirect 2']
    df= df.stack().reset_index()
    df.rename(columns = {'level_0':'Neuron'}, inplace = True)
    df.rename(columns = {'level_1':'Connection'}, inplace = True)
    df.rename(columns = {0:variable}, inplace = True)

    #Bar plots with seaborn
    fig, axes = plt.subplots(nrows= 3,ncols=2,figsize=(20*cm, 30*cm)) # All together
    _fontsize = 6

    cur_df = weigth_df.loc[weigth_df['Connection']== 'Direct']
    _color = 'blue'
    sns.barplot(ax = axes[0,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
                color=_color)
    axes[0,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    cur_df = partners_df.loc[partners_df['Connection']== 'Direct']
    _color = 'blue'
    sns.barplot(ax = axes[0,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
                color=_color)
    axes[0,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    cur_df = weigth_df.loc[weigth_df['Connection']== 'Indirect 1']

    _color = 'orange'
    sns.barplot(ax = axes[1,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
                color=_color)
    axes[1,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    cur_df = partners_df.loc[partners_df['Connection']== 'Indirect 1']
    _color = 'orange'
    sns.barplot(ax = axes[1,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
                color=_color)
    axes[1,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    cur_df = weigth_df.loc[weigth_df['Connection']== 'Indirect 2']
    _color = 'green'
    sns.barplot(ax = axes[2,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
                color=_color)
    axes[2,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    cur_df = partners_df.loc[partners_df['Connection']== 'Indirect 2']
    _color = 'green'
    sns.barplot(ax = axes[2,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
                color=_color)

    axes[2,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

    axes[0,0].set_ylabel('Synaptic count', fontsize = 8)
    axes[0,1].set_ylabel('Partners', fontsize = 8)
    axes[1,0].set_ylabel('Synaptic count', fontsize = 8)
    axes[1,1].set_ylabel('Partners', fontsize = 8)
    axes[2,0].set_ylabel('Synaptic count', fontsize = 8)
    axes[2,1].set_ylabel('Partners', fontsize = 8)

    axes[0,0].spines['right'].set_visible(False)
    axes[0,1].spines['right'].set_visible(False)
    axes[1,0].spines['right'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[2,0].spines['right'].set_visible(False)
    axes[2,1].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,1].spines['top'].set_visible(False)
    axes[1,0].spines['top'].set_visible(False)
    axes[1,1].spines['top'].set_visible(False)
    axes[2,0].spines['top'].set_visible(False)
    axes[2,1].spines['top'].set_visible(False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' - Direct- , - Indirect 1 - and -Indirect 2- connections'
    fig.suptitle(_title, fontsize = 12)

    # Stacked bar plot with pandas
    fig_s, axes_s = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

    weigth_df_bar = weigth_df.groupby(['Neuron','Connection'])[variable_weigth].sum().unstack().fillna(0)
    weigth_df_bar.reindex(user_parameters['defined_microcirtuit']).plot(ax=axes_s[0],kind='bar', stacked=True)
    partners_df_bar = partners_df.groupby(['Neuron','Connection'])[variable_partners].sum().unstack().fillna(0)
    partners_df_bar.reindex(user_parameters['defined_microcirtuit']).plot(ax=axes_s[1],kind='bar', stacked=True)

    axes_s[0].set_ylabel('Synaptic count', fontsize = 8)
    axes_s[1].set_ylabel('Partners', fontsize = 8)

    axes_s[0].spines['right'].set_visible(False)
    axes_s[1].spines['right'].set_visible(False)
    axes_s[0].spines['top'].set_visible(False)
    axes_s[1].spines['top'].set_visible(False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column']
    fig_s.suptitle(_title, fontsize = 12)

    return fig, fig_s


def centrality_plot(centrality_df,user_parameters):#
    '''
    
    '''

    # Barplots using seaborn
    fig, axes = plt.subplots(nrows= 2,ncols=2,figsize=(20*cm, 20*cm)) # All together
    _fontsize = 6

    cur_df = centrality_df

    _color = 'tomato'
    sns.barplot(ax = axes[0,0],x = 'Neuron', y = 'Degree', data = cur_df, order=cur_df.sort_values('Degree',ascending = False).Neuron,
                color=_color)
    axes[0,0].set_xticklabels(cur_df.sort_values('Degree',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


    _color = 'purple'
    sns.barplot(ax = axes[0,1],x = 'Neuron', y = 'Pagerank eigenvector', data = cur_df, order=cur_df.sort_values('Pagerank eigenvector',ascending = False).Neuron,
                color=_color)
    axes[0,1].set_xticklabels(cur_df.sort_values('Pagerank eigenvector',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


    _color = 'teal'
    sns.barplot(ax = axes[1,0],x = 'Neuron', y = 'Betweenness', data = cur_df, order=cur_df.sort_values('Betweenness',ascending = False).Neuron,
                color=_color)
    axes[1,0].set_xticklabels(cur_df.sort_values('Betweenness',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


    _color = 'gold'
    sns.barplot(ax = axes[1,1],x = 'Neuron', y = 'Closeness', data = cur_df, order=cur_df.sort_values('Closeness',ascending = False).Neuron,
                color=_color)
    axes[1,1].set_xticklabels(cur_df.sort_values('Closeness',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


    axes[0,0].spines['right'].set_visible(False)
    axes[0,1].spines['right'].set_visible(False)
    axes[1,0].spines['right'].set_visible(False)
    axes[1,1].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,1].spines['top'].set_visible(False)
    axes[1,0].spines['top'].set_visible(False)
    axes[1,1].spines['top'].set_visible(False)


    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' Centrality measures'
    fig.suptitle(_title, fontsize = 12)

    #plt.show()
    #plt.close()
    return fig
