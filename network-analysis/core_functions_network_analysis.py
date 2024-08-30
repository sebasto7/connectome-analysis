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
import math 

# General variables
cm = 1/2.54  # centimeters in inches

#Initializing the Graph Class
class Graph:
    """
    A class representing a directed graph with nodes and weighted edges.

    This class allows for the creation and manipulation of a graph, including adding nodes and edges, and storing distances (weights) for the edges.

    Attributes:
    nodes (set): A set of nodes in the graph.
    nodes_num (list): A list of node identifiers (numerical values).
    edges (defaultdict(list)): A dictionary mapping nodes to lists of adjacent nodes (outgoing edges).
    distances (dict): A dictionary mapping edge tuples (fromNode, toNode) to their respective weights.
    """

    def __init__(self):
        """
        Initializes an empty graph with no nodes, edges, or distances.
        """
        self.nodes = set()
        self.nodes_num = []
        self.edges = defaultdict(list)
        self.distances = {}

    def addNode(self, value):
        """
        Adds a node to the graph.

        Parameters:
        value (hashable): The value representing the node to be added. Nodes are unique and are stored in a set.
        """
        self.nodes.add(value)

    def addNode_num(self, value):
        """
        Adds a numerical identifier for a node to the graph.

        Parameters:
        value (int): The numerical identifier for the node. This is stored in a list.
        """
        self.nodes_num.append(value)

    def addEdge(self, fromNode, toNode, distance):
        """
        Adds a directed edge to the graph with a specified weight.

        Parameters:
        fromNode (hashable): The starting node of the edge.
        toNode (hashable): The ending node of the edge.
        distance (float): The weight of the edge representing the distance or cost from `fromNode` to `toNode`.

        Notes:
        - If the edge already exists, its weight will be updated.
        - The edge is added to the adjacency list and the distance dictionary.
        """
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance

        


#%% Initial graph functions        

#Implementing Dijkstra's Algorithm

#Single Source Shortest Path (SSSP) - Dijkstra's algorithm
#code source: https://pythonwife.com/dijkstras-algorithm-in-python/

def dijkstra(graph, initial):
    """
    Computes the shortest path from a starting node to all other nodes in a weighted graph using Dijkstra's algorithm.

    This function finds the shortest path from the `initial` node to all other nodes in the graph. It returns both 
    the shortest path distances and the paths themselves.

    Parameters:
    graph (object): A graph object that contains nodes and edges, with edge weights specified in `graph.distances`.
    initial (hashable): The starting node from which to calculate shortest paths.

    Returns:
    tuple: A tuple containing two elements:
        - visited (dict): A dictionary where the keys are nodes and the values are the shortest path distances from the `initial` node.
        - path (defaultdict(list)): A dictionary where the keys are nodes and the values are lists of nodes forming the shortest path to that node.
    """
    visited = {initial: 0}
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


def graph_creation(Table, user_parameters, microcircuit, by_instance=False):
    """
    Creates a graph from a DataFrame of synaptic connections between neurons.

    This function generates a directed graph from a table of synaptic connections. It creates nodes for each neuron 
    and edges for each connection, with weights representing connection strength. The graph is optionally aggregated 
    based on user-defined parameters.

    Parameters:
    Table (pd.DataFrame): DataFrame containing columns specifying presynaptic and postsynaptic neurons, and the connection strength.
    user_parameters (dict): Dictionary of user-defined parameters including neuron naming conventions, aggregation settings, and edge length transformation functions.
    microcircuit (bool): Flag indicating whether to restrict the graph to a specific microcircuit defined in `user_parameters`.
    by_instance (bool, optional): Flag indicating whether to use instance-based neuron naming. Default is `False`.

    Returns:
    tuple: A tuple containing two elements:
        - G (nx.DiGraph): A NetworkX directed graph with nodes and weighted edges.
        - customGraph (Graph): A custom graph object with nodes and edges, including additional attributes.

    Notes:
    - Neurons can be aggregated based on the `user_parameters['aggregate_ii']` setting.
    - Edge weights can be transformed according to the `user_parameters['edge_length_tranformation_function']`.
    - The function creates both a NetworkX graph and a custom graph object for further use.
    """
    if by_instance:
        _pre_neuron_naming = 'instance_pre'
        _post_neuron_naming = 'instance_post'
    else:
        _pre_neuron_naming = 'PreSynapticNeuron'
        _post_neuron_naming = 'PostSynapticNeuron'

    # Creating list of nodes
    presynaptic_neurons = Table[_pre_neuron_naming].unique()
    postsynaptic_neurons = Table[_post_neuron_naming].unique()

    print(f'\nGraph created with: \n Presynaptic neurons: {presynaptic_neurons} \n Postsynaptic neurons: {postsynaptic_neurons}\n')

    if microcircuit:
        neurons = user_parameters['defined_microcircuit']
        Table = Table[(Table['PreSynapticNeuron'].isin(neurons)) & (Table['PostSynapticNeuron'].isin(neurons))].copy()
    else:
        neurons = list(np.unique(np.concatenate((presynaptic_neurons, postsynaptic_neurons), axis=0)))
    for i, n in enumerate(neurons):
        if n[-4:] == 'home':
            neurons[i] = n[0:-4]

    neurons_list_aggregated = []
    for neuron in neurons:
        temp_neuron = neuron[0:user_parameters['aggregate_ii']]
        if temp_neuron not in neurons_list_aggregated:
            neurons_list_aggregated.append(temp_neuron)
    user_parameters['neurons_list_aggregated'] = neurons_list_aggregated

    # Initializing and filling the graph with nodes and edges
    customGraph = Graph()
    neuron_list = neurons_list_aggregated if user_parameters['aggregate'] else neurons

    for n, neuron in enumerate(neuron_list):
        customGraph.addNode(neuron)
        customGraph.addNode_num(n)

    for index, row in Table.iterrows():
        if user_parameters['aggregate']:
            pair = row[_pre_neuron_naming][0:user_parameters['aggregate_ii']], row[_post_neuron_naming][0:user_parameters['aggregate_ii']]
        else:
            _pre = row[_pre_neuron_naming]
            _post = row[_post_neuron_naming]
            pair = _pre, _post

        if pair in customGraph.distances:
            temp_N = customGraph.distances[pair] + row['N']
            customGraph.addEdge(pair[0], pair[1], temp_N)
        else:
            customGraph.addEdge(pair[0], pair[1], row['N'])

    # Applying length transformation functions
    Weights = customGraph.distances
    for key, value in Weights.items():
        Weights[key] = round(value)

    if user_parameters['edge_length_tranformation_function'] == 'reciprocal_function':
        edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()]
    elif user_parameters['edge_length_tranformation_function'] == 'linear_flip_function':
        edges = [(k[0], k[1], {'weight': (-v + int(max(Table['N'])) + 1)}) for k, v in Weights.items()]
    elif user_parameters['edge_length_tranformation_function'] == 'binary':
        edges = [(k[0], k[1], {'weight': 1}) for k, v in Weights.items()]
    elif user_parameters['edge_length_tranformation_function'] == 'none':
        edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    return G, customGraph



#%% Plotting functions

def node_to_node_graph_analysis_and_plot(G, Weights, user_parameters, dirPath, save_figures, start_node_ls, last_node_ls):
    """
    Analyzes and visualizes all possible paths between pairs of nodes in a network graph.

    Parameters:
    -----------
    G : networkx.Graph
        The graph representing the network.
    Weights : dict
        Dictionary containing edge weights, where keys are tuples of (node1, node2) 
        and values are the corresponding weights.
    user_parameters : dict
        Dictionary containing user-defined parameters such as cutoff for path length, 
        graph type, and whether to plot paths.
    dirPath : str
        Directory path for saving figures if `save_figures` is True.
    save_figures : bool
        If True, save the generated figures in the specified directory.
    start_node_ls : list
        List of starting nodes for path analysis.
    last_node_ls : list
        List of ending nodes for path analysis.

    Returns:
    --------
    path_df : pandas.DataFrame
        A DataFrame containing information about the analyzed paths, including start 
        node, end node, path, weight, normalized weight, and number of jumps.
    """
    multiple_path_dict = {}  # For storing paths between multiple start and end nodes

    for last_node in last_node_ls:
        for start_node in start_node_ls:
            path_list = []  # Paths between a single start node and last node
            for path in nx.all_simple_paths(G, source=start_node, target=last_node, cutoff=user_parameters['_cutoff']):
                path_list.append(path)
            _key = start_node + '_' + last_node
            multiple_path_dict[_key] = path_list

    path_df = pd.DataFrame(columns=['Start_node', 'Last_node', 'Node_to_node', 'Path', 'Weigth'])
    for key, value in multiple_path_dict.items():
        if not value:  # Skip if there are no paths
            continue

        node_to_node_list = [key] * len(value)
        start_node_list = [key.split('_')[0]] * len(value)
        last_node_list = [key.split('_')[1]] * len(value)
        path_list = value
        path_weight_list = []

        for cur_path in path_list:
            highlighted_path = cur_path
            temp_weight = 0
            if len(highlighted_path) > 1:
                for i in range(len(highlighted_path) - 1):
                    temp_weight += Weights[(highlighted_path[i], highlighted_path[i + 1])]
                weight_path = round(temp_weight / (len(highlighted_path) - 1))
                path_weight_list.append(weight_path)
            else:
                continue

            if user_parameters['plot_node_to_tode_paths']:
                path_fig, axes = plt.subplots(figsize=(20 * cm, 20 * cm))
                if highlighted_path[0] == 'L1-':
                    _color = '#E64D00'
                elif highlighted_path[0] == 'L2-':
                    _color = '#8019E8'
                elif highlighted_path[0] == 'L3-':
                    _color = '#1B9E77'
                else:
                    _color = 'r'

                path_edges = list(zip(highlighted_path, highlighted_path[1:]))
                pos = nx.circular_layout(G)

                nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes) - set(highlighted_path), node_size=600)
                nx.draw_networkx_edges(G, pos, edgelist=set(G.edges) - set(path_edges), width=2)
                nx.draw_networkx_nodes(G, pos, nodelist=highlighted_path, node_color=_color, node_size=900)
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=_color, width=8)
                nx.draw_networkx_labels(G, pos)

                path_fig.suptitle(f'Node-to-node path. Averaged weight: {weight_path:.1f}  {user_parameters["graph"]}')
                plt.axis('equal')

                if save_figures:
                    name_nodes = f"{user_parameters['start_node']} to {user_parameters['last_node']}"
                    save_dir = os.path.join(dirPath, 'figures', 'paths', name_nodes)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if len(highlighted_path) == 3:
                        path_fig.savefig(os.path.join(save_dir, f'weight {weight_path:.1f} path {user_parameters["start_node"]}-{highlighted_path[1]}-{user_parameters["last_node"]} {user_parameters["column"]} {user_parameters["graph"]}.pdf'), bbox_inches="tight")
                    elif len(highlighted_path) == 4:
                        path_fig.savefig(os.path.join(save_dir, f'weight {weight_path:.1f} path {user_parameters["start_node"]}-{highlighted_path[1]}-{highlighted_path[2]}-{user_parameters["last_node"]} {user_parameters["column"]} {user_parameters["graph"]}.pdf'), bbox_inches="tight")

        cur_df = pd.DataFrame(list(zip(start_node_list, last_node_list, node_to_node_list, path_list, path_weight_list)),
                              columns=['Start_node', 'Last_node', 'Node_to_node', 'Path', 'Weigth'])
        path_df = path_df._append(cur_df)

    path_df['Norm_weigth'] = path_df.apply(lambda row: row['Weigth'] / (len(row['Path']) - 1), axis=1)  # Normalize weight by path length
    path_df['Jumps'] = path_df['Path'].apply(lambda x: len(x) - 1)  # Number of jumps between nodes

    print(f'Node to node path analysis done. Total number of paths: {len(path_df)}')
    plt.close()
    return path_df


def mean_path_length_plot(path_df, user_parameters, save_figures):
    """
    Analyzes and plots the mean path length statistics for the paths between nodes.

    Parameters:
    -----------
    path_df : pandas.DataFrame
        DataFrame containing the analyzed paths, with columns for start node, end node, 
        path, weight, normalized weight, and jumps.
    user_parameters : dict
        Dictionary containing user-defined parameters.
    save_figures : bool
        If True, save the generated figures.

    Returns:
    --------
    None
    """
    agg_functions = ['sum', 'mean', 'median', 'min', 'max', 'std', 'sem', 'count', 'describe', 'size', 'first', 'last']
    mean_path_df = path_df.groupby(['Start_node', 'Last_node']).agg(agg_functions)

    return print('Under construction')  # Placeholder for future implementation


def distribution_path_length_plot(path_df, user_parameters, save_figures):
    """
    Generates histograms to visualize the distribution and cumulative distribution of 
    path lengths and weights in the analyzed network paths.

    Parameters:
    -----------
    path_df : pandas.DataFrame
        DataFrame containing the analyzed paths, with columns for start node, end node, 
        path, weight, normalized weight, and jumps.
    user_parameters : dict
        Dictionary containing user-defined parameters.
    save_figures : bool
        If True, save the generated figures.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the generated histograms.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30 * cm, 30 * cm))

    sns.histplot(x=path_df['Weigth'], stat='probability', ax=axes[0])
    axes[0].set_ylabel('Probability')
    axes[0].set_xlabel('Weight')

    sns.histplot(x=path_df['Norm_weigth'], stat='probability', ax=axes[1])
    axes[1].set_ylabel('Probability')
    axes[1].set_xlabel('Normalized Weight')

    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    plt.close()
    return fig


def path_length_transformation_plot(Table, user_parameters, transformation_function):
    """
    Generates a set of line plots that visualize the effect of a chosen transformation 
    function on path lengths within a network.

    Parameters:
    -----------
    Table : pandas.DataFrame
        DataFrame containing the network data, where the 'N' column represents the 
        number of connections for each node.
    user_parameters : dict
        Dictionary containing user-defined parameters such as graph type and column 
        names for title customization.
    transformation_function : str
        The name of the transformation function to apply. Options include:
        - 'reciprocal_function': Applies the reciprocal function y = 1/x to transform 
          the data.
        - 'linear_flip_function': Applies a linear flip function to reverse and adjust 
          the data linearly.
        - 'none': No transformation is applied.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the generated plots.
    """
    # Line plot path length transformation

    x = list(range(1, len(Table.N) + 1))
    y = list(Table.N)  # Connections

    if transformation_function == "reciprocal_function":
        new_y = [1 / i for i in y]  # 1/x function -> new distances
        new_x = [1 / i for i in x]  # 1/x function
        y_des = sorted(y, reverse=True)
        y_as = sorted(y, reverse=False)
        new_y_des = sorted(new_y, reverse=True)
        new_y_as = sorted(new_y, reverse=False)
        new_x_as = sorted(new_x, reverse=False)

        d = {}  # Getting frequency of connections
        for i in y:
            d[i] = d.get(i, 0) + 1
        import operator
        d_des = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True))
        y_freq = list(d_des.values())
        y_freq_norm = [float(i) / max(y_freq) for i in y_freq]
        x_rank = range(1, len(y_freq) + 1)

        # Linear regression
        m, c = np.polyfit(np.log10(x_rank), np.log10(y_freq), 1, w=np.sqrt(y_freq))  # fit log(y) = m*log(x) + c
        y_freq_fit = np.exp(m * np.log10(x_rank) + c)  # calculate the fitted values of y

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30 * cm, 30 * cm))
        _title = f"Edge definition as the reciprocal function, {user_parameters['graph']} : {user_parameters['column']}"
        fig.suptitle(_title, fontsize=12)

        axes[0, 0].plot(y_as, new_y_des)  # Edge distance transformation
        axes[0, 0].set_ylabel('Edge (distance)', fontsize=12)
        axes[0, 0].set_xlabel('Number of connections', fontsize=12)

        axes[1, 0].plot(y_as, new_y_des)  # Power law
        axes[1, 0].set_ylabel('Edge (distance)', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('Number of connections', fontsize=12)

        axes[0, 1].scatter(x_rank, y_freq, s=30, alpha=0.7, edgecolors="k")  # Zipf law connections
        axes[0, 1].set_xlabel('Connection rank', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)

        axes[1, 1].scatter(x_rank, y_freq, s=30, alpha=0.7, edgecolors="k")  # Zipf law connections
        axes[1, 1].set_xlabel('Connection rank', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xscale('log')

        axes[0, 0].spines['right'].set_visible(False)
        axes[0, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        axes[1, 0].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        axes[0, 1].spines['top'].set_visible(False)

    elif transformation_function == "linear_flip_function":
        new_y = [(-i + int(max(y)) + 1) for i in y]  # linear flip function -> new distances
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30 * cm, 30 * cm))
        _title = f"Edge definition linearly flipped, {user_parameters['graph']} : {user_parameters['column']}"
        fig.suptitle(_title, fontsize=12)

        axes[0, 0].plot(y)  # Original distances
        axes[0, 0].set_ylabel('Edge (distance)', fontsize=12)
        axes[0, 0].set_xlabel('Connections', fontsize=12)

        axes[0, 1].plot(x, new_y)  # Transformed distances
        axes[0, 1].set_ylabel('Edge (distance) flipped', fontsize=12)
        axes[0, 1].set_xlabel('Connections', fontsize=12)

        sns.histplot(x=y, stat='count', binwidth=1, ax=axes[1, 0])
        axes[1, 0].set_ylabel('Counts', fontsize=12)
        axes[1, 0].set_xlabel('Synaptic count (weight)', fontsize=12)

        sns.histplot(x=new_y, stat='count', binwidth=1, ax=axes[1, 1])
        axes[1, 1].set_ylabel('Counts', fontsize=12)
        axes[1, 1].set_xlabel('Synaptic count (weight) flipped', fontsize=12)

    print("Line plots for path length transformation done.")
    plt.close()
    return fig


def connections_histomgram_plot(Table, user_parameters):
    """
    Generates histograms that visualize the distribution and cumulative distribution 
    of the number of connections in the network.

    Parameters:
    -----------
    Table : pandas.DataFrame
        DataFrame containing the network data, where the 'N' column represents the 
        number of connections for each node.
    user_parameters : dict
        Dictionary containing user-defined parameters such as graph type and column 
        names for title customization.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the generated histograms.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30 * cm, 30 * cm))
    axes[0].hist(list(Table.N), bins=50, density=True)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_xlabel('Number of connections', fontsize=12)

    axes[1].hist(list(Table.N), bins=50, cumulative=True, density=True)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel('Cumulative', fontsize=12)
    axes[1].set_xlabel('Number of connections', fontsize=12)

    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    _title = f"Number of connections, {user_parameters['graph']} : {user_parameters['column']}"
    fig.suptitle(_title, fontsize=12)
    plt.close()

    return fig


def graph_plot(Weights, user_parameters, transformation_function):
    """
    Generates a network graph with edges weighted according to a chosen transformation 
    function. The graph is plotted using NetworkX with different layout options.

    Parameters:
    -----------
    Weights : dict
        Dictionary containing edge weights between nodes, where keys are tuples of 
        (node1, node2) and values are the corresponding weights.
    user_parameters : dict
        Dictionary containing user-defined parameters such as graph type, neuron 
        list, and column names for title customization.
    transformation_function : str
        The name of the transformation function to apply to edge weights. Options include:
        - 'reciprocal_function': Applies the reciprocal function y = 1/x to transform 
          the data.
        - 'linear_flip_function': Applies a linear flip function to reverse and adjust 
          the data linearly.
        - 'binary': Converts all edge weights to 1.
        - 'none': No transformation is applied.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the generated network graph.
    """
    concentricGraph = False # To generate a graph based on layers


    # Determine the min and max edge values for normalization
    min_weight = min(Weights.values())
    max_weight = max(Weights.values())

    if min_weight == max_weight:
        # Avoid division by zero when all edge weights are the same
        min_weight = 0

    # Normalize the edge values to the range [0, 1]
    normalized_weights = {k: (v - min_weight) / (max_weight - min_weight) for k, v in Weights.items()}

    if transformation_function  == 'reciprocal_function':
        edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()]  ## The min-max normalizatino can be applied here die to division with 0
    elif transformation_function  == "linear_flip_function":
        edges = [(k[0], k[1], {'weight': (-v+1)}) for k, v in normalized_weights.items()] 
    elif transformation_function  == "binary":
        edges = [(k[0], k[1], {'weight': 1}) for k, v in normalized_weights.items()] 
    elif transformation_function == "none":
        edges = [(k[0], k[1], {'weight': v}) for k, v in normalized_weights.items()]


    # # Create a dictionary to store the positions of edges between the same nodes
    # edge_positions = {}

    # for u, v, d in edges:
    #     if (u, v) not in edge_positions and (v, u) not in edge_positions:
    #         # If this is the first edge between these nodes, position it at (0, 0)
    #         edge_positions[(u, v)] = [(0, 0)]
    #     elif (u, v) in edge_positions:
    #         # If there's already an edge from u to v, add a slight y-shift
    #         edge_positions[(u, v)].append((0, len(edge_positions[(u, v)]) * 0.05))
    #     else:
    #         # If there's already an edge from v to u, add a slight x-shift
    #         edge_positions[(v, u)].append((len(edge_positions[(v, u)]) * 0.05, 0))

    # # Modify the positions of edges based on the calculated edge positions
    # pos_edges = []
    # for u, v, d in edges:
    #     if (u, v) in edge_positions:
    #         pos_edges.append((u, v, {'weight': d['weight'], 'pos': edge_positions[(u, v)].pop(0)}))
    #     elif (v, u) in edge_positions:
    #         pos_edges.append((u, v, {'weight': d['weight'], 'pos': edge_positions[(v, u)].pop(0)}))



    ## Old version without normalization
    # if transformation_function  == 'reciprocal_function':
    #     edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()] 
    # elif transformation_function  == "linear_flip_function":
    #     edges = [(k[0], k[1], {'weight': (-v+int(max(Weights.values()))+1)}) for k, v in Weights.items()] 
    # elif transformation_function  == "binary":
    #     edges = [(k[0], k[1], {'weight': 1}) for k, v in Weights.items()] 

    # elif transformation_function == "none":
    #     edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()]

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
        #pos = nx.spring_layout(G, pos = initial_node_pos, fixed = initial_node_pos ) # positions for all nodes
        #pos = nx.random_layout(G)
        pos = nx.circular_layout(G)
        #pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G) 
        #pos = nx.spiral_layout(G)


    fig,axes= plt.subplots(figsize=(20*cm, 20*cm))
    fig.suptitle(f"Column {user_parameters['column']}, edge-length transformation: {transformation_function}")

    ## NODES
    nx.draw_networkx_nodes(G,pos,node_size=300)

    ## LABELS
    nx.draw_networkx_labels(G,pos,font_size=6,font_family='sans-serif')

    ## EDGES
    #nx.draw_networkx_edges(G,pos,edgelist=edges, width=2,connectionstyle="arc3,rad=-0.2")
    #nx.draw_networkx_edges(G,pos,edgelist=edges, width=2) # all edges with same width

    
    # Modify the nx.draw_networkx_edges function to use the normalized edge values for width
    widths = [d['weight']*8 for u, v, d in G.edges(data=True)]
    #nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths*2)#
    nx.draw_networkx_edges(G, pos, width=widths,connectionstyle="arc3,rad=-0.2")


    ## Some tests
    # Draw the networkx edges with modified positions
    # for u, v, d in pos_edges:
    #     pos_u = pos[u]
    #     pos_v = pos[v]
    #     pos_edge = [(pos_u[0] + d['pos'][0], pos_u[1] + d['pos'][1]),
    #                 (pos_v[0] + d['pos'][0], pos_v[1] + d['pos'][1])]
    #     nx.draw_networkx_edges(G, edgelist=[(u, v)], width=d['weight'], pos=pos_edge, ax = axes)

    # # Draw the networkx edges with modified positions
    # for u, v, d in pos_edges:
    #     pos = nx.circular_layout(G) 
    #     pos_u = pos[u]
    #     pos_v = pos[v]

    #     #Shifting some positions
    #     pos_u[0] = pos_u[0] + d['pos'][0]
    #     pos_u[1] = pos_u[1] + d['pos'][1]

    #     pos_v[0] = pos_u[0] + d['pos'][0]
    #     pos_v[1] = pos_u[1] + d['pos'][1]

    #     print([(u, v)])

    #     nx.draw_networkx_edges(G, pos={u: pos_u, v: pos_v}, edgelist=[(u, v)], width=d['weight'], ax=axes)






    ## weights
    labels = nx.get_edge_attributes(G,'weight')
    for key, value in labels.items():
        if transformation_function  == 'reciprocal_function':
            labels [key]= round(1/value)
        elif transformation_function  == "linear_flip_function":
            labels [key]= (-value+int(max(Weights.values()))+1)
        elif transformation_function  == "binary":
            labels [key]= value
        else:
            break
    #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,label_pos = 0.35,font_size=5)
    plt.close()

    return fig

def input_output_plot(node_of_interest, final_input_output_dict, final_input_df, final_output_df, 
                      final_input_ranked_df, final_output_ranked_df, final_input_ranked_norm_df, 
                      final_output_ranked_norm_df, user_parameters):
    """
    Generates various plots to visualize the input and output connections for a specific neuron.

    Parameters:
    node_of_interest (str): The neuron to be analyzed and plotted.
    final_input_output_dict (dict): Dictionary containing input-output DataFrames for each neuron.
    final_input_df (pd.DataFrame): DataFrame containing all input connections for the analyzed neurons.
    final_output_df (pd.DataFrame): DataFrame containing all output connections for the analyzed neurons.
    final_input_ranked_df (pd.DataFrame): DataFrame containing ranked input connections.
    final_output_ranked_df (pd.DataFrame): DataFrame containing ranked output connections.
    final_input_ranked_norm_df (pd.DataFrame): DataFrame containing normalized ranked input connections.
    final_output_ranked_norm_df (pd.DataFrame): DataFrame containing normalized ranked output connections.
    user_parameters (dict): A dictionary containing user-defined parameters for plotting, such as neurons to exclude.

    Returns:
    tuple: Contains the generated figures:
        - fig (Figure): Bar plots showing absolute number of input and output synapses.
        - fig_s (Figure): Stacked bar plots showing input/output synapses per neuron type.
        - fig_r (Figure): Stacked bar plots showing input/output synapses for the top N connections.
        - fig_f (Figure): Stacked bar plots showing normalized input/output synapses as fractions.
        - fig_l (Figure): Line plots for the first N connections showing fractions of input/output synapses.

    """

    ####################### Bar plots with seaborn  ######################
    # Input and output absolute number of connections (bar) and in percentage as text

    # Getting the data for the node_of_interest
    input_df = final_input_output_dict[node_of_interest]['input_neurons']
    output_df = final_input_output_dict[node_of_interest]['output_neurons']

    input_percentage = (input_df['Inputs'] / sum(input_df['Inputs'])) * 100
    input_df['Percentage'] = input_percentage.round(2)
    output_percentage = (output_df['Outputs'] / sum(output_df['Outputs'])) * 100
    output_df['Percentage'] = output_percentage.round(2)

    # Bar plots with seaborn
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30 * cm, 20 * cm))  # All together
    cur_df = input_df
    _color = 'blue'
    sns.barplot(ax=axes[0], x='Neuron', y='Inputs', data=cur_df, 
                order=cur_df.sort_values('Inputs', ascending=False).Neuron, color=_color)

    axes[0].set_xticklabels(cur_df.sort_values('Inputs', ascending=False).Neuron.tolist(), rotation=90, fontsize=7)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    autolabel(axes[0], cur_df['Percentage'], ascending=False)

    cur_df = output_df
    _color = 'green'
    sns.barplot(ax=axes[1], x='Neuron', y='Outputs', data=cur_df, 
                order=cur_df.sort_values('Outputs', ascending=False).Neuron, color=_color)
    axes[1].set_xticklabels(cur_df.sort_values('Outputs', ascending=False).Neuron.tolist(), rotation=90, fontsize=7)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    autolabel(axes[1], cur_df['Percentage'], ascending=False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' - ' + user_parameters['node_of_interest']
    fig.suptitle(_title, fontsize=12)

    #################### Stacked bar plot with pandas  ###################
    # Inputs and outputs synapses per neuron type
    _fontsize = 6
    fig_s, axes_s = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))  # All together

    weigth_df_bar = final_input_df
    weigth_df_bar.plot(ax=axes_s[0], kind='bar', stacked=True, fontsize=_fontsize)
    weigth_df_bar = final_output_df
    weigth_df_bar.plot(ax=axes_s[1], kind='bar', stacked=True, fontsize=_fontsize)

    axes_s[0].set_ylabel('Input synapses per neuron type', fontsize=12)
    axes_s[1].set_ylabel('Output synapses per neuron type', fontsize=12)

    axes_s[0].spines['right'].set_visible(False)
    axes_s[1].spines['right'].set_visible(False)
    axes_s[0].spines['top'].set_visible(False)
    axes_s[1].spines['top'].set_visible(False)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by synaptic partner'
    fig_s.suptitle(_title, fontsize=12)

    #################### Stacked bar plot with pandas  ###################
    # Inputs and outputs synapses per neuron, RANKED, first "N" connections
    
    N = 20
    fig_r, axes_r = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))  # All together

    weigth_df_bar = final_input_ranked_df.iloc[:, :N]
    weigth_df_bar.plot(ax=axes_r[0], kind='bar', stacked=True, fontsize=_fontsize)
    weigth_df_bar = final_output_ranked_df.iloc[:, :N]
    weigth_df_bar.plot(ax=axes_r[1], kind='bar', stacked=True, fontsize=_fontsize)

    axes_r[0].set_ylabel('Input synapses, ranked', fontsize=12)
    axes_r[1].set_ylabel('Output synapses, ranked', fontsize=12)

    axes_r[0].spines['right'].set_visible(False)
    axes_r[1].spines['right'].set_visible(False)
    axes_r[0].spines['top'].set_visible(False)
    axes_r[1].spines['top'].set_visible(False)

    axes_r[0].get_legend().remove()
    axes_r[1].get_legend().remove()

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by connection number'
    fig_r.suptitle(_title, fontsize=12)

    #################### Stacked bar plot with pandas  ###################
    # Normalized to the sum of input. Input-output FRACTIONS

    fig_f, axes_f = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))  # All together

    weigth_df_bar = final_input_ranked_norm_df.iloc[:, :N]
    weigth_df_bar.plot(ax=axes_f[0], kind='bar', stacked=True, fontsize=_fontsize)
    weigth_df_bar = final_output_ranked_norm_df.iloc[:, :N]
    weigth_df_bar.plot(ax=axes_f[1], kind='bar', stacked=True, fontsize=_fontsize)

    axes_f[0].set_ylabel('Fraction of input synapses', fontsize=12)
    axes_f[1].set_ylabel('Fraction of output synapses', fontsize=12)

    axes_f[0].spines['right'].set_visible(False)
    axes_f[1].spines['right'].set_visible(False)
    axes_f[0].spines['top'].set_visible(False)
    axes_f[1].spines['top'].set_visible(False)

    axes_f[0].get_legend().remove()
    axes_f[1].get_legend().remove()

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' FRACTIONS'
    fig_f.suptitle(_title, fontsize=12)

    ############################ Line plots with pandas ######################
    # First "N" connections in absolute numbers

    N = 7
    fig_l, axes_l = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))  # All together

    weigth_line_df = final_input_ranked_norm_df.fillna(0)
    weigth_line_df = weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
    weigth_line_df.T.plot(ax=axes_l[0])

    weigth_line_df = final_output_ranked_norm_df.fillna(0)
    weigth_line_df = weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
    weigth_line_df.T.plot(ax=axes_l[1])

    axes_l[0].set_ylabel('Fraction of input synapses', fontsize=_fontsize)
    axes_l[1].set_ylabel('Fraction of output synapses', fontsize=_fontsize)

    axes_l[0].spines['right'].set_visible(False)
    axes_l[1].spines['right'].set_visible(False)
    axes_l[0].spines['top'].set_visible(False)
    axes_l[1].spines['top'].set_visible(False)
    axes_l[0].set_ylim(0, 1)
    axes_l[1].set_ylim(0, 1)

    _title = user_parameters['graph'] + ': ' + user_parameters['column'] + ': ' + user_parameters['node_of_interest'] + ' FRACTIONS'
    fig_l.suptitle(_title, fontsize=12)
    
    plt.close(fig)
    plt.close(fig_s)
    plt.close(fig_r)
    plt.close(fig_f)
    plt.close(fig_l)

    return fig, fig_s, fig_r, fig_f, fig_l


def direct_indirect_connections_plot(number_partners_dict, length_dict, norm_length_dict, user_parameters):
    """
    Generates bar plots and stacked bar plots to visualize direct and indirect connections for neurons.

    This function creates plots to compare the number of direct and indirect partners and the corresponding weights
    of connections for neurons. It also generates stacked bar plots to summarize the connection weights and partner counts.

    Parameters:
    number_partners_dict (dict): Dictionary where keys are neuron names and values are lists containing counts of direct,
                                 indirect 1, and indirect 2 connections.
    length_dict (dict): Dictionary where keys are neuron names and values are lists containing the weights of direct,
                        indirect 1, and indirect 2 connections.
    norm_length_dict (dict): Dictionary where keys are neuron names and values are lists containing the normalized
                              weights of direct, indirect 1, and indirect 2 connections.
    user_parameters (dict): Dictionary containing user-defined parameters for plotting, such as graph and column names.

    Returns:
    tuple: Contains the generated figures:
        - fig (Figure): Bar plots showing absolute numbers and weights of direct and indirect connections.
        - fig_s (Figure): Stacked bar plots showing the summed weights and partner counts for each neuron.

    """
    # Create DataFrames from dictionaries
    variable_partners = '# of Partners'
    partners_df = pd.DataFrame(number_partners_dict).T
    partners_df.columns = ['Direct', 'Indirect 1', 'Indirect 2']
    partners_df = partners_df.stack().reset_index()
    partners_df.rename(columns={'level_0': 'Neuron', 'level_1': 'Connection', 0: variable_partners}, inplace=True)

    variable_weight = 'Weight'
    weight_df = pd.DataFrame(length_dict).T
    weight_df.columns = ['Direct', 'Indirect 1', 'Indirect 2']
    weight_df = weight_df.stack().reset_index()
    weight_df.rename(columns={'level_0': 'Neuron', 'level_1': 'Connection', 0: variable_weight}, inplace=True)

    variable = 'Norm Weight'
    df = pd.DataFrame(norm_length_dict).T
    df.columns = ['Direct', 'Indirect 1', 'Indirect 2']
    df = df.stack().reset_index()
    df.rename(columns={'level_0': 'Neuron', 'level_1': 'Connection', 0: variable}, inplace=True)

    # Bar plots with seaborn
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20 * cm, 30 * cm))
    _fontsize = 6

    for i, conn_type in enumerate(['Direct', 'Indirect 1', 'Indirect 2']):
        cur_weight_df = weight_df.loc[weight_df['Connection'] == conn_type]
        cur_partners_df = partners_df.loc[partners_df['Connection'] == conn_type]
        
        _color = ['blue', 'orange', 'green'][i]
        sns.barplot(ax=axes[i, 0], x='Neuron', y='Weight', data=cur_weight_df,
                    order=cur_weight_df.sort_values('Weight', ascending=False).Neuron, color=_color)
        axes[i, 0].set_xticklabels(cur_weight_df.sort_values('Weight', ascending=False).Neuron.tolist(), rotation=90, fontsize=_fontsize)

        sns.barplot(ax=axes[i, 1], x='Neuron', y='# of Partners', data=cur_partners_df,
                    order=cur_partners_df.sort_values('# of Partners', ascending=False).Neuron, color=_color)
        axes[i, 1].set_xticklabels(cur_partners_df.sort_values('# of Partners', ascending=False).Neuron.tolist(), rotation=90, fontsize=_fontsize)

        axes[i, 0].set_ylabel('Synaptic Count', fontsize=8)
        axes[i, 1].set_ylabel('Partners', fontsize=8)
        axes[i, 0].spines['right'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 1].spines['top'].set_visible(False)

    _title = f"{user_parameters['graph']}: {user_parameters['column']} - Direct, Indirect 1, and Indirect 2 Connections"
    fig.suptitle(_title, fontsize=12)

    # Stacked bar plot with pandas
    fig_s, axes_s = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))

    weight_df_bar = weight_df.groupby(['Neuron', 'Connection'])['Weight'].sum().unstack().fillna(0)
    weight_df_bar.reindex(user_parameters['defined_microcircuit']).plot(ax=axes_s[0], kind='bar', stacked=True)
    partners_df_bar = partners_df.groupby(['Neuron', 'Connection'])['# of Partners'].sum().unstack().fillna(0)
    partners_df_bar.reindex(user_parameters['defined_microcircuit']).plot(ax=axes_s[1], kind='bar', stacked=True)

    axes_s[0].set_ylabel('Synaptic Count', fontsize=8)
    axes_s[1].set_ylabel('Partners', fontsize=8)
    axes_s[0].spines['right'].set_visible(False)
    axes_s[1].spines['right'].set_visible(False)
    axes_s[0].spines['top'].set_visible(False)
    axes_s[1].spines['top'].set_visible(False)

    _title = f"{user_parameters['graph']}: {user_parameters['column']}"
    fig_s.suptitle(_title, fontsize=12)

    plt.close()
    plt.close()

    return fig, fig_s

def centrality_plot(centrality_df, user_parameters):
    """
    Generates bar plots to visualize centrality measures for neurons.

    This function creates bar plots to display various centrality measures (Degree, Pagerank eigenvector, Betweenness,
    and Closeness) for neurons.

    Parameters:
    centrality_df (pd.DataFrame): DataFrame where rows represent neurons and columns represent different centrality measures.
    user_parameters (dict): Dictionary containing user-defined parameters for plotting, such as graph and column names.

    Returns:
    Figure: Contains bar plots for Degree, Pagerank eigenvector, Betweenness, and Closeness centrality measures.

    """
    # Bar plots using seaborn
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20 * cm, 20 * cm))
    _fontsize = 6

    for i, (measure, color) in enumerate([('Degree', 'tomato'), 
                                           ('Pagerank', 'purple'),
                                           ('Betweenness', 'teal'),
                                           ('Closeness', 'gold')]):
        row, col = divmod(i, 2)
        sns.barplot(ax=axes[row, col], x='Neuron', y=measure, data=centrality_df,
                    order=centrality_df.sort_values(measure, ascending=False).Neuron, color=color)
        axes[row, col].set_xticklabels(centrality_df.sort_values(measure, ascending=False).Neuron.tolist(), rotation=90, fontsize=_fontsize)
        axes[row, col].spines['right'].set_visible(False)
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].set_ylabel(measure, fontsize=8)

    _title = f"{user_parameters['graph']}: {user_parameters['column']} Centrality Measures"
    fig.suptitle(_title, fontsize=12)
    plt.close()

    return fig

def heatmap_plot(short_col_names, df, list_of_neurons, user_parameters, data_name):
    """
    Generates heatmaps to visualize input/output variability for a list of neurons.

    This function creates three types of heatmaps: absolute counts, normalized to the maximum count, and normalized
    to the sum of counts. The heatmaps visualize connections from/to each neuron.

    Parameters:
    short_col_names (list of str): List of column names to be used as y-axis labels in heatmaps.
    df (pd.DataFrame): DataFrame where the index represents neurons and columns represent pre- or postsynaptic partners.
    list_of_neurons (list of str): List of neuron names to be visualized in heatmaps.
    user_parameters (dict): Dictionary containing user-defined parameters for plotting, such as graph and column names.
    data_name (str): Description of the data being visualized (e.g., 'inputs' or 'outputs').

    Returns:
    tuple: Contains the generated heatmap figures:
        - fig (Figure): Heatmaps of absolute connection counts.
        - fig_max (Figure): Heatmaps of connections normalized to the maximum count per neuron.
        - fig_sum (Figure): Heatmaps of connections normalized to the sum of counts per neuron.

    """
    # Color palettes
    _palette = sns.color_palette("viridis", as_cmap=True)

    # First figure: absolute connection counts
    ncols = 2
    nrows = math.ceil(len(list_of_neurons) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 20 * cm, nrows * 20 * cm))
    fig.suptitle(f"{user_parameters['graph']}: {user_parameters['column']} {data_name} Heatmap", fontsize=12)

    for n, neuron in enumerate(list_of_neurons):
        if n >= len(axes.flatten()):
            break
        _data = df.loc[neuron].dropna(axis='columns', how='all')
        sns.heatmap(cmap=_palette, ax=axes.flatten()[n], data=_data, square=True, cbar=False)
        cax = plt.colorbar(axes.flatten()[n].collections[0], ax=axes.flatten()[n], orientation="vertical", shrink=0.5)
        cax.ax.tick_params(labelsize=6)
        axes.flatten()[n].tick_params(axis='x', labelrotation=90, labelsize=6)
        axes.flatten()[n].set_xlabel(f'{data_name} neuron')
        axes.flatten()[n].set_ylabel('Columns')
        axes.flatten()[n].set_yticklabels(short_col_names)
        axes.flatten()[n].set_xticks(range(len(_data.columns)))
        axes.flatten()[n].set_xticklabels(_data.columns)
        axes.flatten()[n].set_title(f'{neuron}')

    # Second figure: connections normalized to the maximum count per row
    df_max = df.div(df.max(axis=1), axis=0).copy()
    fig_max, axes_max = plt.subplots(nrows, ncols, figsize=(ncols * 20 * cm, nrows * 20 * cm))
    fig_max.suptitle(f"{user_parameters['graph']}: {user_parameters['column']} {data_name} Heatmap Normalized to Max", fontsize=12)

    for n, neuron in enumerate(list_of_neurons):
        if n >= len(axes_max.flatten()):
            break
        _data = df_max.loc[neuron].dropna(axis='columns', how='all')
        sns.heatmap(cmap=_palette, ax=axes_max.flatten()[n], data=_data, square=True, cbar=False)
        cax = plt.colorbar(axes_max.flatten()[n].collections[0], ax=axes_max.flatten()[n], orientation="vertical", shrink=0.5)
        axes_max.flatten()[n].tick_params(axis='x', labelrotation=90, labelsize=6)
        axes_max.flatten()[n].set_xlabel(f'{data_name} neuron')
        axes_max.flatten()[n].set_ylabel('Columns')
        axes_max.flatten()[n].set_yticklabels(short_col_names)
        axes_max.flatten()[n].set_xticks(range(len(_data.columns)))
        axes_max.flatten()[n].set_xticklabels(_data.columns)
        axes_max.flatten()[n].set_title(f'{neuron}')

    # Third figure: connections normalized to the sum of counts per row
    df_sum = df.div(df.sum(axis=1), axis=0).copy()
    fig_sum, axes_sum = plt.subplots(nrows, ncols, figsize=(ncols * 20 * cm, nrows * 20 * cm))
    fig_sum.suptitle(f"{user_parameters['graph']}: {user_parameters['column']} {data_name} Heatmap Normalized to Sum", fontsize=12)

    for n, neuron in enumerate(list_of_neurons):
        if n >= len(axes_sum.flatten()):
            break
        _data = df_sum.loc[neuron].dropna(axis='columns', how='all')
        sns.heatmap(cmap=_palette, ax=axes_sum.flatten()[n], data=_data, square=True, cbar=False)
        cax = plt.colorbar(axes_sum.flatten()[n].collections[0], ax=axes_sum.flatten()[n], orientation="vertical", shrink=0.5)
        axes_sum.flatten()[n].tick_params(axis='x', labelrotation=90, labelsize=4)
        axes_sum.flatten()[n].set_xlabel(f'{data_name} neuron')
        axes_sum.flatten()[n].set_ylabel('Columns')
        axes_sum.flatten()[n].set_yticklabels(short_col_names)
        axes_sum.flatten()[n].set_xticks(range(len(_data.columns)))
        axes_sum.flatten()[n].set_xticklabels(_data.columns)
        axes_sum.flatten()[n].set_title(f'{neuron}')

    plt.close()
    plt.close()
    plt.close()

    return fig, fig_max, fig_sum


#%% Analysis functions    

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def centrality_analysis(G, user_parameters):
    """
    Performs centrality analysis on a given directed graph using various centrality measures.

    Parameters:
    G (networkx.DiGraph): A directed graph representing neurons and their connections.
    user_parameters (dict): A dictionary containing user-defined parameters such as neurons to exclude.

    Returns:
    pd.DataFrame: A DataFrame containing various centrality measures for each neuron, excluding specified neurons.
    """

    from networkx.algorithms.centrality import (degree_centrality, in_degree_centrality, out_degree_centrality,
                                                closeness_centrality, betweenness_centrality, eigenvector_centrality)

    # Calculate centrality measures
    degree_dict = degree_centrality(G)
    in_degree_dict = in_degree_centrality(G)
    out_degree_dict = out_degree_centrality(G)
    eigenvector_dict = eigenvector_centrality(G)
    pagerank_dict = nx.pagerank(G, alpha=0.8)
    betweenness_dict = betweenness_centrality(G, normalized=True, weight="weight")
    closeness_dict = closeness_centrality(G, distance="weight", wf_improved=True)

    # Convert centrality dictionaries to DataFrames
    degree_df = pd.DataFrame(degree_dict, index=[0]).T
    in_degree_df = pd.DataFrame(in_degree_dict, index=[0]).T
    out_degree_df = pd.DataFrame(out_degree_dict, index=[0]).T
    eigenvector_df = pd.DataFrame(eigenvector_dict, index=[0]).T
    pagerank_df = pd.DataFrame(pagerank_dict, index=[0]).T
    betweenness_df = pd.DataFrame(betweenness_dict, index=[0]).T
    closeness_df = pd.DataFrame(closeness_dict, index=[0]).T

    # Combine all centrality measures into a single DataFrame
    centrality_df = pd.concat([degree_df, in_degree_df, out_degree_df, eigenvector_df, pagerank_df, betweenness_df, closeness_df], axis=1)
    centrality_df.columns = ['Degree', 'IN-Degree', 'OUT-Degree', 'Eigenvector', 'Pagerank', 'Betweenness', 'Closeness']
    centrality_df.reset_index(level=0, inplace=True)
    centrality_df.rename(columns={'index': 'Neuron'}, inplace=True)

    # Exclude specified neurons
    centrality_df = centrality_df[~centrality_df['Neuron'].isin(user_parameters['neurons_to_exclude'])]

    print('Centrality measures analysis done.')
    plt.close()
    return centrality_df

def direct_indirect_connections_analysis(Weights, user_parameters):
    """
    Analyzes direct and indirect connections for all neurons in a directed graph.

    Parameters:
    Weights (dict): A dictionary where keys are neuron pairs and values are connection weights.
    user_parameters (dict): A dictionary containing user-defined parameters such as the list of neurons to analyze and those to exclude.

    Returns:
    tuple: A tuple containing dictionaries for the number of synaptic partners, total connection lengths, and normalized connection lengths.
    """

    interconnections = 3  # Number of direct and indirect connections to assess
    number_partners_dict = {}  # Number of synaptic partners for each type of interconnection
    length_dict = {}  # Total length of connections
    norm_length_dict = {}  # Normalized length to the number of contact partners

    # Create directed graph from weights
    edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    for neuron in user_parameters['neuron_list']:
        if neuron in user_parameters['neurons_to_exclude']:
            continue
        
        # Calculate shortest paths from the neuron using Dijkstra's algorithm
        length, path = nx.single_source_dijkstra(G, neuron)
        
        temp_connections_list = []
        temp_total_length_list = []
        temp_total_norm_length_list = []

        for i in range(interconnections):
            temp_total_length = 0
            temp_number_of_partners = 0

            for key in path.keys():
                if len(path[key]) == 2 + i:
                    temp_number_of_partners += 1
                    temp_total_length += length[key]

            temp_connections_list.append(temp_number_of_partners)
            temp_total_length_list.append(temp_total_length)

            if temp_number_of_partners == 0:
                temp_total_norm_length_list.append(None)
            else:
                temp_total_norm_length_list.append(temp_total_length / temp_number_of_partners)

        number_partners_dict[neuron] = temp_connections_list
        length_dict[neuron] = temp_total_length_list
        norm_length_dict[neuron] = temp_total_norm_length_list

    print('Direct and indirect connections analysis done.')
    plt.close()
    return number_partners_dict, length_dict, norm_length_dict

def input_output_analysis(Weights, neuron_ls):
    """
    Performs input-output analysis for each neuron in a list, focusing on synaptic connections.

    Parameters:
    Weights (dict): A dictionary where keys are neuron pairs and values are connection weights.
    neuron_ls (list): A list of neurons to be analyzed.

    Returns:
    tuple: A tuple containing dictionaries and DataFrames with input-output data, ranked data, and normalized ranked data.
    """

    edges = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()]

    final_input_output_dict = {}
    final_input_df = pd.DataFrame()
    final_output_df = pd.DataFrame()
    final_input_ranked_df = pd.DataFrame()
    final_output_ranked_df = pd.DataFrame()

    for neuron in neuron_ls:
        input_neurons_dict = {}
        output_neurons_dict = {}
        temp_input_output_dict = {}

        for item in edges:
            input_neuron = item[0]
            output_neuron = item[1]
            weight = float(item[2]['weight'])
            
            if input_neuron == neuron:
                output_neurons_dict[output_neuron] = weight
            if output_neuron == neuron:
                input_neurons_dict[input_neuron] = weight

        # Sort and convert dictionaries to DataFrames
        input_neurons_df = pd.DataFrame(input_neurons_dict, index=[0]).T.rename(columns={0: 'Inputs'}).reset_index().rename(columns={'index': 'Neuron'})
        output_neurons_df = pd.DataFrame(output_neurons_dict, index=[0]).T.rename(columns={0: 'Outputs'}).reset_index().rename(columns={'index': 'Neuron'})

        temp_input_output_dict['input_neurons'] = input_neurons_df
        temp_input_output_dict['output_neurons'] = output_neurons_df
        final_input_output_dict[neuron] = temp_input_output_dict

        # Append to final input/output DataFrames
        final_input_df = final_input_df._append(pd.DataFrame(input_neurons_dict, index=[0]).rename(index={0: neuron}))
        final_output_df = final_output_df._append(pd.DataFrame(output_neurons_dict, index=[0]).rename(index={0: neuron}))

        # Rank and normalize data
        temp_input_ranked_df = final_input_df.loc[neuron].sort_values(ascending=False).to_frame().T
        temp_output_ranked_df = final_output_df.loc[neuron].sort_values(ascending=False).to_frame().T

        for i, column in enumerate(temp_input_ranked_df.columns):
            temp_input_ranked_df.rename(columns={column: i + 1}, inplace=True)
        for i, column in enumerate(temp_output_ranked_df.columns):
            temp_output_ranked_df.rename(columns={column: i + 1}, inplace=True)

        final_input_ranked_df = final_input_ranked_df._append(temp_input_ranked_df)
        final_output_ranked_df = final_output_ranked_df._append(temp_output_ranked_df)

    # Normalize the ranked DataFrames
    final_input_ranked_norm_df = final_input_ranked_df.div(final_input_ranked_df.sum(axis=1), axis=0)
    final_output_ranked_norm_df = final_output_ranked_df.div(final_output_ranked_df.sum(axis=1), axis=0)

    print('Input-Output analysis done.')
    return final_input_output_dict, final_input_df, final_output_df, final_input_ranked_df, final_output_ranked_df, final_input_ranked_norm_df, final_output_ranked_norm_df

def autolabel(ax, variable, ascending=True):
    """
    Annotates bar plot bars with text labels showing percentages.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to plot on.
    variable (pd.Series): The variable containing percentage values to annotate.
    ascending (bool): If True, sorts the values in ascending order; otherwise, descending.
    """

    rects = ax.patches
    if not ascending:
        variable = variable.sort_values(ascending=False).to_list()
    for i, rect in enumerate(rects):
        height = rect.get_height()
        percent = f'{variable[i]} %'
        ax.text(rect.get_x() + rect.get_width() / 2., 0.2 * height,
                percent, ha='center', va='bottom', rotation=90, color='k')



    

