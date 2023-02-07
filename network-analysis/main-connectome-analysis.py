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
from core_functions_network_analysis import path_length_transformation_plot, connections_histomgram_plot, graph_plot,node_to_node_graph_analysis_and_plot
from core_functions_network_analysis import Graph, centrality_analysis, direct_indirect_connections_analysis, input_output_analysis
from core_functions_general import saveWorkspace


#%% Messages to developer


#%% User parameters in a dictionary
user_parameters = {}
user_parameters['dataPath']= r'D:\Connectomics-Data\datasets\NeuPrint\fib25'
user_parameters['file']='F_column_fib25.csv'#dataset file
user_parameters['column']='F column'
user_parameters['graph']= 'Fib25_data_7medulla_columns'
user_parameters['aggregate']= False # For merging nodes with the same begginning of name of length = aggregate_ii
user_parameters['aggregate_ii']=4
user_parameters['start_node']= 'L3' # string of lenght = aggregate_ii
user_parameters['last_node']= 'Tm9'  # string of lenght = aggregate_ii
user_parameters['node_of_interest']= 'Tm9'  # for input-output plotting
user_parameters['_cutoff']=3 # max num of neurons between start and last node
user_parameters['neurons_to_exclude']= ['L1','L2','L3','T4a','T4b','T4c','T4d'] # exclude from centrality analysis and SSSP Analysis, not from the graph
user_parameters['multiple_start_nodes']= ['L1','L2','L3']
user_parameters['synaptic_stength_filter'] = 0
user_parameters['defined_microcirtuit'] = ['Tm1','Tm2','Tm4','Tm9','CT1','C3', 'C2', 'T1'] # Used so far only for the stacked bar plot



#%% Other user parameters:
save_data = 1
save_figures = 1
plot_node_to_tode_paths = 0
main_data_folder = r'D:\Connectomics-Data'
split_columns = False # True only for data files where the info of all columns is mixed, code not finished

#%% Useful variables
cm = 1/2.54  # centimeters in inches

#%% Auto creation of important paths
dirPath = os.path.join(main_data_folder, user_parameters['column'], user_parameters['graph'])
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
main_processed_data_folder = os.path.join(main_data_folder,'Processed data all',user_parameters['graph'])




# Loading the data of the circuit graph - from csv file
filePath = os.path.join(user_parameters['dataPath'], user_parameters['file'])
Table = pd.read_csv(filePath)
Table = Table[Table['N'] >user_parameters['synaptic_stength_filter']]
if split_columns:
    Table = Table[Table.PreSynapticNeuron.str.endswith(user_parameters['column'][0])]
    Table = Table[Table.PostSynapticNeuron.str.endswith(user_parameters['column'][0])]


#%% Data visualization 
#Plotting path length transformations:
fig_path_lentgh_transform = path_length_transformation_plot(Table, user_parameters, 'reciprocal_function')

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

edges = [(k[0], k[1], {'weight': 1/v}) for k, v in Weights.items()] # Applied transformation
G = nx.DiGraph() 
#G = nx.Graph()
G.add_edges_from(edges)

# temporary untill fixing some code below
edges_plot = [(k[0], k[1], {'weight': v}) for k, v in Weights.items()] 
G_plot = nx.DiGraph()
G_plot.add_edges_from(edges_plot)
    
#%% Visualizing the graph
fig_graph = graph_plot(Weights, user_parameters, 'none')


#%% Single source shortest path  (SSSP) analysis
# with NetwrokX
length, path = nx.single_source_dijkstra(G, user_parameters['start_node'])

path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items())
message_str = '\n >>>>>>> All paths: \n\n%s can reach a neuron via [shortest path]: \n\n%s ' % (user_parameters['start_node'], path_str)
print(message_str)

path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items() if k == user_parameters['last_node'])
message_str = '\n >>>>>>> Single path: \n\n%s can reach %s via [shortest path]: \n\n%s ' % (user_parameters['start_node'],user_parameters['last_node'], path_str)
print(message_str)


#%% Node to node analysis
#Highlights specific paths in the graph between two nodes and save a dataFrame with all the paths
path_df = node_to_node_graph_analysis_and_plot(G, Weights, user_parameters,dirPath,save_figures,plot_node_to_tode_paths)    

#%% Centrality measures analysis
#Currently extracting: closeness_centrality, betweenness_centrality, degree_centrality, and eigenvector_centrality
centrality_df = centrality_analysis(G,user_parameters)


#%% Direct and indirect connections analysis for all nodes
number_partners_dict, length_dict, norm_length_dict = direct_indirect_connections_analysis(Weights,user_parameters)

#%% Input- ouput analysis
final_input_output_dict, final_input_df,final_output_df, final_input_ranked_df ,final_output_ranked_df = input_output_analysis(Weights,user_parameters)

    
  
      
#%% Pandas dataframes creation  (Seb, move to a function)
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

#%% Plotting centrality measures

############################ Bar plots with seaborn ##########################

# Barplots
bar_fig_c, bar_axes_c = plt.subplots(nrows= 2,ncols=2,figsize=(20*cm, 20*cm)) # All together
_fontsize = 6

cur_df = centrality_df

_color = 'tomato'
sns.barplot(ax = bar_axes_c[0,0],x = 'Neuron', y = 'Degree', data = cur_df, order=cur_df.sort_values('Degree',ascending = False).Neuron,
            color=_color)
bar_axes_c[0,0].set_xticklabels(cur_df.sort_values('Degree',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


_color = 'purple'
sns.barplot(ax = bar_axes_c[0,1],x = 'Neuron', y = 'Pagerank eigenvector', data = cur_df, order=cur_df.sort_values('Pagerank eigenvector',ascending = False).Neuron,
            color=_color)
bar_axes_c[0,1].set_xticklabels(cur_df.sort_values('Pagerank eigenvector',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


_color = 'teal'
sns.barplot(ax = bar_axes_c[1,0],x = 'Neuron', y = 'Betweenness', data = cur_df, order=cur_df.sort_values('Betweenness',ascending = False).Neuron,
            color=_color)
bar_axes_c[1,0].set_xticklabels(cur_df.sort_values('Betweenness',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


_color = 'gold'
sns.barplot(ax = bar_axes_c[1,1],x = 'Neuron', y = 'Closeness', data = cur_df, order=cur_df.sort_values('Closeness',ascending = False).Neuron,
            color=_color)
bar_axes_c[1,1].set_xticklabels(cur_df.sort_values('Closeness',ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)


bar_axes_c[0,0].spines['right'].set_visible(False)
bar_axes_c[0,1].spines['right'].set_visible(False)
bar_axes_c[1,0].spines['right'].set_visible(False)
bar_axes_c[1,1].spines['right'].set_visible(False)
bar_axes_c[0,0].spines['top'].set_visible(False)
bar_axes_c[0,1].spines['top'].set_visible(False)
bar_axes_c[1,0].spines['top'].set_visible(False)
bar_axes_c[1,1].spines['top'].set_visible(False)


_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' Centrality measures'
bar_fig_c.suptitle(_title, fontsize = 12)

plt.show()
plt.close()

#%% Plotting synaptic count (weigth) and number of partners


############################ Stacked bar plot with pandas######################

# First barplot
bar_fig, bar_axes = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

weigth_df_bar = weigth_df.groupby(['Neuron','Connection'])[variable_weigth].sum().unstack().fillna(0)
weigth_df_bar.reindex(user_parameters['defined_microcirtuit']).plot(ax=bar_axes[0],kind='bar', stacked=True)
partners_df_bar = partners_df.groupby(['Neuron','Connection'])[variable_partners].sum().unstack().fillna(0)
partners_df_bar.reindex(user_parameters['defined_microcirtuit']).plot(ax=bar_axes[1],kind='bar', stacked=True)

bar_axes[0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes[1].set_ylabel('Partners', fontsize = 8)

bar_axes[0].spines['right'].set_visible(False)
bar_axes[1].spines['right'].set_visible(False)
bar_axes[0].spines['top'].set_visible(False)
bar_axes[1].spines['top'].set_visible(False)

_title = user_parameters['graph'] + ': ' + user_parameters['column']
bar_fig.suptitle(_title, fontsize = 12)
plt.show()
plt.close()


############################ Bar plots with seaborn ##########################


# Second barplot
bar_fig2, bar_axes2 = plt.subplots(nrows= 3,ncols=2,figsize=(20*cm, 30*cm)) # All together
_fontsize = 6

cur_df = weigth_df.loc[weigth_df['Connection']== 'Direct']
_color = 'blue'
sns.barplot(ax = bar_axes2[0,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
            color=_color)
bar_axes2[0,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

cur_df = partners_df.loc[partners_df['Connection']== 'Direct']
_color = 'blue'
sns.barplot(ax = bar_axes2[0,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
            color=_color)
bar_axes2[0,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

cur_df = weigth_df.loc[weigth_df['Connection']== 'Indirect 1']

_color = 'orange'
sns.barplot(ax = bar_axes2[1,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
            color=_color)
bar_axes2[1,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

cur_df = partners_df.loc[partners_df['Connection']== 'Indirect 1']
_color = 'orange'
sns.barplot(ax = bar_axes2[1,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
            color=_color)
bar_axes2[1,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

cur_df = weigth_df.loc[weigth_df['Connection']== 'Indirect 2']
_color = 'green'
sns.barplot(ax = bar_axes2[2,0],x = 'Neuron', y = variable_weigth, data = cur_df, order=cur_df.sort_values(variable_weigth,ascending = False).Neuron,
            color=_color)
bar_axes2[2,0].set_xticklabels(cur_df.sort_values(variable_weigth,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

cur_df = partners_df.loc[partners_df['Connection']== 'Indirect 2']
_color = 'green'
sns.barplot(ax = bar_axes2[2,1],x = 'Neuron', y = variable_partners, data = cur_df, order=cur_df.sort_values(variable_partners,ascending = False).Neuron,
            color=_color)

bar_axes2[2,1].set_xticklabels(cur_df.sort_values(variable_partners,ascending = False).Neuron.tolist(), rotation = 90, fontsize = _fontsize)

bar_axes2[0,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[0,1].set_ylabel('Partners', fontsize = 8)
bar_axes2[1,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[1,1].set_ylabel('Partners', fontsize = 8)
bar_axes2[2,0].set_ylabel('Synaptic count', fontsize = 8)
bar_axes2[2,1].set_ylabel('Partners', fontsize = 8)

bar_axes2[0,0].spines['right'].set_visible(False)
bar_axes2[0,1].spines['right'].set_visible(False)
bar_axes2[1,0].spines['right'].set_visible(False)
bar_axes2[1,1].spines['right'].set_visible(False)
bar_axes2[2,0].spines['right'].set_visible(False)
bar_axes2[2,1].spines['right'].set_visible(False)
bar_axes2[0,0].spines['top'].set_visible(False)
bar_axes2[0,1].spines['top'].set_visible(False)
bar_axes2[1,0].spines['top'].set_visible(False)
bar_axes2[1,1].spines['top'].set_visible(False)
bar_axes2[2,0].spines['top'].set_visible(False)
bar_axes2[2,1].spines['top'].set_visible(False)




_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' - Direct- , - Indirect 1 - and -Indirect 2- connections'
bar_fig2.suptitle(_title, fontsize = 12)
plt.show()
plt.close()

#%% Input- ouput plotting for a node_of_interest
    
node_of_interest = user_parameters['node_of_interest']
input_df = final_input_output_dict[node_of_interest]['input_neurons']
output_df = final_input_output_dict[node_of_interest]['output_neurons']

input_percentatge = (input_df['Inputs']/sum(input_df['Inputs']))*100
input_df['Percentage'] = input_percentatge
input_df['Percentage']= input_df['Percentage'].round(2)
output_percentatge = (output_df['Outputs']/sum(output_df['Outputs']))*100
output_df['Percentage'] = output_percentatge
output_df['Percentage']= output_df['Percentage'].round(2)

############################ Bar plots with seaborn ##########################

def autolabel(ax, variable, ascending = True):
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

# Third Barplot
bar_fig3, bar_axes3 = plt.subplots(nrows= 2,ncols=1,figsize=(30*cm, 20*cm)) # All together
cur_df = input_df
_color = 'blue'
sns.barplot(ax = bar_axes3[0],x = 'Neuron', y = 'Inputs', data = cur_df, order=cur_df.sort_values('Inputs',ascending = False).Neuron,
            color=_color)

bar_axes3[0].set_xticklabels(cur_df.sort_values('Inputs',ascending = False).Neuron.tolist(), rotation = 90, fontsize = 7)
bar_axes3[0].spines['right'].set_visible(False)
bar_axes3[0].spines['top'].set_visible(False)
autolabel(bar_axes3[0],cur_df['Percentage'],ascending = False)


cur_df = output_df
_color = 'green'
sns.barplot(ax = bar_axes3[1],x = 'Neuron', y = 'Outputs', data = cur_df, order=cur_df.sort_values('Outputs',ascending = False).Neuron,
            color=_color)
bar_axes3[1].set_xticklabels(cur_df.sort_values('Outputs',ascending = False).Neuron.tolist(), rotation = 90, fontsize = 7)       
bar_axes3[1].spines['right'].set_visible(False)
bar_axes3[1].spines['top'].set_visible(False)
autolabel(bar_axes3[1],cur_df['Percentage'],ascending = False)

_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' - '+ user_parameters['node_of_interest']
bar_fig3.suptitle(_title, fontsize = 12)

plt.show()
plt.close()  

#%% Inputs and outputs synapses per neuron, color codes for all neuron types

############################ Stacked bar plot with pandas######################

# Fifth barplot
_fontsize = 6
bar_fig5, bar_axes5 = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

weigth_df_bar = final_input_df
#weigth_df_bar= weigth_df_bar.fillna(0)
weigth_df_bar.plot(ax=bar_axes5[0],kind='bar', stacked=True, fontsize = _fontsize)
weigth_df_bar = final_output_df
#weigth_df_bar= weigth_df_bar.fillna(0)
weigth_df_bar.plot(ax=bar_axes5[1],kind='bar', stacked=True, fontsize = _fontsize)


bar_axes5[0].set_ylabel('Input synapses per neuron type', fontsize = 12)
bar_axes5[1].set_ylabel('Output synapses per neuron type', fontsize = 12)

bar_axes5[0].spines['right'].set_visible(False)
bar_axes5[1].spines['right'].set_visible(False)
bar_axes5[0].spines['top'].set_visible(False)
bar_axes5[1].spines['top'].set_visible(False)

_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by synaptic partner'
bar_fig5.suptitle(_title, fontsize = 12)
plt.show()
plt.close()

#%% Inputs and outputs synapses per neuron, RANKED, first "N" connections

############################ Stacked bar plot with pandas######################

N = 20

# Sixth barplot
_fontsize = 6
bar_fig6, bar_axes6 = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together

weigth_df_bar = final_input_ranked_df.iloc[: , :N]
weigth_df_bar.plot(ax=bar_axes6[0],kind='bar', stacked=True, fontsize = _fontsize)
weigth_df_bar = final_output_ranked_df.iloc[: , :N]
weigth_df_bar.plot(ax=bar_axes6[1],kind='bar', stacked=True, fontsize = _fontsize)


bar_axes6[0].set_ylabel('Input synapses, ranked', fontsize = 12)
bar_axes6[1].set_ylabel('Output synapses, ranked', fontsize = 12)

bar_axes6[0].spines['right'].set_visible(False)
bar_axes6[1].spines['right'].set_visible(False)
bar_axes6[0].spines['top'].set_visible(False)
bar_axes6[1].spines['top'].set_visible(False)

bar_axes6[0].get_legend().remove()
bar_axes6[1].get_legend().remove()

_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' by connection number'
bar_fig6.suptitle(_title, fontsize = 12)
plt.show()
plt.close()

# Normalized to the sum of inputes (plotting input-output fractions)

# Seventh barplot
bar_fig7, bar_axes7 = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together
_fontsize = 6

weigth_df_bar = final_input_ranked_norm_df.iloc[: , :N]
weigth_df_bar.plot(ax=bar_axes7[0],kind='bar', stacked=True, fontsize = _fontsize)
weigth_df_bar = final_output_ranked_norm_df.iloc[: , :N]
weigth_df_bar.plot(ax=bar_axes7[1],kind='bar', stacked=True, fontsize = _fontsize)


bar_axes7[0].set_ylabel('Fraction of input synapses', fontsize = 12)
bar_axes7[1].set_ylabel('Fraction of output synapses', fontsize = 12)

bar_axes7[0].spines['right'].set_visible(False)
bar_axes7[1].spines['right'].set_visible(False)
bar_axes7[0].spines['top'].set_visible(False)
bar_axes7[1].spines['top'].set_visible(False)

bar_axes7[0].get_legend().remove()
bar_axes7[1].get_legend().remove()

_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ' FRACTIONS'
bar_fig7.suptitle(_title, fontsize = 12)
plt.show()
plt.close()


############################ Line plots with pandas######################

N = 7

#Firt line plot
line_fig1, line_axes1 = plt.subplots(nrows= 1,ncols=2,figsize=(20*cm, 10*cm)) # All together
_fontsize = 6
weigth_line_df= final_input_ranked_norm_df.fillna(0)
weigth_line_df=weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
weigth_line_df.T.plot(ax=line_axes1[0])

weigth_line_df= final_output_ranked_norm_df.fillna(0)
weigth_line_df=weigth_line_df.loc[user_parameters['node_of_interest']].iloc[:N]
weigth_line_df.T.plot(ax=line_axes1[1])

line_axes1[0].set_ylabel('Fraction of input synapses', fontsize = _fontsize)
line_axes1[1].set_ylabel('Fraction of output synapses', fontsize = _fontsize)

line_axes1[0].spines['right'].set_visible(False)
line_axes1[1].spines['right'].set_visible(False)
line_axes1[0].spines['top'].set_visible(False)
line_axes1[1].spines['top'].set_visible(False)


_title = user_parameters['graph'] + ': ' + user_parameters['column'] + ': ' + user_parameters['node_of_interest'] +' FRACTIONS'
line_fig1.suptitle(_title, fontsize = 12)
plt.show()
plt.close()


#%% Saving data
if save_figures:
        
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    save_dir = dirPath +'\\figures\\'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) # Seb: creating figures folder     
    graph_fig.savefig(save_dir+'Graph %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")        
    bar_fig.savefig(save_dir+'Stacked bar plot %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    bar_fig2.savefig(save_dir+'Bar plots %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    bar_fig_c.savefig(save_dir+'Centrality measures %s %s.pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    bar_fig3.savefig(save_dir+'Inputs and outputs - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    bar_fig7.savefig(save_dir+'Inputs and outputs - FRACTION  %s %s .pdf' % (user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")
    line_fig1.savefig(save_dir+'Inputs and outputs FRACTION - %s %s %s .pdf' % (user_parameters['node_of_interest'], user_parameters['column'],user_parameters['graph']),bbox_inches = "tight")

    print('\nFigures saved\n')
    
if save_data:
    os.chdir(main_data_folder) # Seb: X_save_vars.txt file needs to be there    
    varDict = locals()
    pckl_save_name = ('%s' % user_parameters['column'])
    saveOutputDir = dirPath+ '\\processed data\\' 
    if not os.path.exists(saveOutputDir): # Seb: creating processed data folder
        os.mkdir(saveOutputDir) 
    saveWorkspace(saveOutputDir,pckl_save_name, varDict,
                  varFile='main_connectome_analysis_save_vars.txt',extension='.pickle')
    if not os.path.exists(main_processed_data_folder):
        os.mkdir(main_processed_data_folder) # Seb: creating folder for storing all data per stim type
    saveWorkspace(main_processed_data_folder,pckl_save_name, varDict, 
                    varFile='main_connectome_analysis_save_vars.txt',extension='.pickle')
        
    print('\n\n%s processed data saved...\n\n' % pckl_save_name)

       

#%% With custom class and function (currently connected) 

#from core_functions_and_classes_network_analysis import dijkstra    
#length, path = dijkstra(customGraph, SSSP_start_node)

# path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items())
# message_str = '>>>>>>> All paths: \n\n%s can reach a neuron via [neuron]: \n\n%s ' % (SSSP_start_node, path_str)
# print(message_str)

# path_str = ''.join('%s: %s ; ' % (k,v) for k,v in path.items() if k == SSSP_last_node)
# message_str = '>>>>>>> Single path: \n\n%s can reach %s via [neuron]: \n\n%s ' % (SSSP_start_node,SSSP_last_node, path_str)
# print(message_str)






