#
# - Clustering of Tm9 synaptic connectivity data using hamming distance - 
#
#%% Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
import matplotlib.gridspec as gridspec
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import hamming
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from sklearn.metrics import silhouette_score
from fafbseg import flywire
import navis


#%% Input parameters
scale_data = False

exclude_neurons = True
neurons_to_exclude = ["L3", "Mi4", "CT1"]

#%% Load the data
main_path = '/Users/burakgur/Documents/Science/flywire-paper'
fig_save_path = os.path.join(main_path,"burak_figures","Hamming_clustering")

current_data = 'Tm9_FAFB_L_R__relative.csv' 
# current_data = 'Tm9_abs_counts_L_5percent_threshold.csv' 

filePath =  os.path.join(main_path,'Tm9_rel_abs_counts',current_data)

data_df = pd.read_csv(filePath, header=0, index_col=0)
data_df = data_df.fillna(0)
order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
         'C3','Tm1','PS125','L4','ML1','TmY17','C2',
         'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
data_df = data_df[order]

if exclude_neurons:
    cluster_df = data_df.drop(columns=neurons_to_exclude)
else:
    cluster_df = data_df

#%% Scale the data (do not do it for binary data)
data_array = cluster_df.to_numpy()
if scale_data:
    data_array[np.isnan(data_array)] = 0

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_array)
    data_array = scaled_features.copy()
    
#%% Hamming distances
# Calculate pairwise Hamming distances
binary_array = (data_array>0).astype(int)

distances = pairwise_distances(binary_array, metric="hamming")

# Perform hierarchical clustering using linkage
linkage_matrix = hierarchy.linkage(distances, method='ward')  # You can choose different linkage methods


# Reorder data based on dendrogram leaves
reordered_indices = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
reordered_array = data_array[reordered_indices]
reordered_distances = pairwise_distances(reordered_array, metric="hamming")

# Create a figure with custom grid layout
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[8, 1], height_ratios=[1.2, 8, 0.5])

# Plot the dendrogram_cosine
ax_dendrogram_hamming = plt.subplot(gs[0, :-1])
ax_dendrogram_hamming.spines['top'].set_visible(False)
ax_dendrogram_hamming.spines['right'].set_visible(False)
ax_dendrogram_hamming.spines['bottom'].set_visible(False)
ax_dendrogram_hamming.spines['left'].set_visible(False)
ax_dendrogram_hamming.get_xaxis().set_visible(False)
ax_dendrogram_hamming.get_yaxis().set_visible(False)
hierarchy.dendrogram(linkage_matrix, ax=ax_dendrogram_hamming, color_threshold=0)

# Plot the heatmap using the reordered DataFrame
ax_heatmap = plt.subplot(gs[1, :-1])
sns.heatmap(reordered_distances, cmap='rocket_r', annot=False, xticklabels=cluster_df.index, yticklabels=cluster_df.index, ax=ax_heatmap, cbar=False)

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
cbar.set_label('Hamming distance')

#%%
sns.clustermap(distances,cmap='rocket_r')
# sns.clustermap(data_array,metric='cosine')

#%% Find out how many clusters could make sense
fig1, axs = plt.subplots(1)
# A list holds the SSE values for each k
# sse = []
max_c = 21
# for k in range(1, max_c):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     sse.append(kmeans.inertia_)
    
# axs[0].plot(range(1, max_c), sse)
# axs[0].set_xticks(range(1, max_c))
# axs[0].set_xlabel("Number of Clusters")
# axs[0].set_ylabel("SSE")
   
# Silhu
silhouette_coefficients = []
   
# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, max_c):
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')

    cluster_labels = clusterer.fit_predict(distances)
    silhouette_coefficients.append(silhouette_score(distances, cluster_labels))

start_plot_k = 6
axs.plot(range(2, max_c), silhouette_coefficients)
# axs[1].set_xticks(range(2, max_c))
axs.set_xlabel("Number of Clusters")
axs.set_ylabel("Silhouette Coefficient")
axs.set_xlim([2,20])

# %% Clustering 
selected_cluster_n = 7
clusterer = AgglomerativeClustering(n_clusters=selected_cluster_n, linkage='ward')
cluster_labels = clusterer.fit_predict(distances)

unique_clusters , counts = np.unique(cluster_labels, return_counts= True)


# %%
# neurons_to_exclude = ["L3", "Mi4", "CT1"]
# plot_df = cluster_df.drop(columns=neurons_to_exclude)
plot_df = cluster_df
# plot_df = cluster_df[order[3:]]
binarized_df = plot_df.applymap(lambda x: 1 if x > 0 else 0)
fig2, axs2 = plt.subplots(selected_cluster_n,figsize=(6, 12))
for icluster, cluster_n in enumerate(unique_clusters):
    current_c_df = binarized_df.iloc[cluster_labels==cluster_n]
    sns.heatmap(current_c_df,ax=axs2[icluster],cbar=False)
    
    axs2[icluster].set_title(f"Cluster{cluster_n}, n:{len(current_c_df)}")
    axs2[icluster].set_yticklabels([])
    if not((icluster == len(unique_clusters)-1)):
        axs2[icluster].set_xticklabels([])
        
fig3, axs3 = plt.subplots(selected_cluster_n,figsize=(5, 15))

for icluster, cluster_n in enumerate(unique_clusters):
    current_c_df = binarized_df.iloc[cluster_labels==cluster_n]    
    new_df = current_c_df.sum()/current_c_df.shape[0]
    perc_df = pd.DataFrame(columns=new_df.index)
    perc_df.loc[len(perc_df)] = new_df.values
    
    sns.barplot(perc_df, ax = axs3[icluster])
    
    axs3[icluster].set_title(f"Cluster{cluster_n}, n:{len(current_c_df)}")
    axs3[icluster].set_ylim([0,1])
    axs3[icluster].set_yticklabels([])
    axs3[icluster].set_xticklabels(axs3[icluster].get_xticklabels(), rotation=90)
    if not((icluster == len(unique_clusters)-1)):
        axs3[icluster].set_xticklabels([])

#%% Save figures
fig.savefig(os.path.join(fig_save_path,f'HeatMapHamming_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}.pdf'))
fig1.savefig(os.path.join(fig_save_path,f'Silhouette_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}.pdf'))
fig2.savefig(os.path.join(fig_save_path,f'BinaryHeatMap_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}.pdf'))
fig3.savefig(os.path.join(fig_save_path,f'InputPercentage_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}.pdf'))



#%% Visualize the clusters
filePath =  os.path.join(main_path,"Tm9 proofreadings - Hoja 1.csv")
location_df = pd.read_csv(filePath, header=0)

OL_labels = [string.split(":")[2] for string in data_df.index]
ids= location_df["optic_lobe_id"]


xyz = np.zeros([len(OL_labels),3])
for idx, neuron in enumerate(OL_labels):
    df_loc = np.where(ids==neuron)[0]
    coordinate = location_df.iloc[df_loc]["XYZ-ME"].to_numpy(dtype=str, copy=True)
    xyz[idx,:] = np.array([coordinate[0].split(',')],dtype=float)
xyz *=[4,4,40] # For plotting it using navis

# %% Plot location 1
# fig4, axs4 = plt.subplots(selected_cluster_n,figsize=(16, 16))

for icluster, cluster_n in enumerate(unique_clusters):
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.scatter(xyz[cluster_labels==cluster_n,0],xyz[cluster_labels==cluster_n,1],
            xyz[cluster_labels==cluster_n,2],'.',color=[0,0,0,0.7], label=f"Cluster {cluster_n}")
    OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
    OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

    ax.azim=200
    ax.elav=20
    ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
    ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
    ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
    # navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    fig.suptitle(f"Cluster{cluster_n} OL L")
    fig.savefig(os.path.join(fig_save_path,f'Location_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}_cluster{cluster_n}_OL-L.pdf'))

for icluster, cluster_n in enumerate(unique_clusters):
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.scatter(xyz[cluster_labels==cluster_n,0],xyz[cluster_labels==cluster_n,1],
            xyz[cluster_labels==cluster_n,2],'.',color=[0,0,0,0.7], label=f"Cluster {cluster_n}")
    OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
    OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

    ax.azim=0
    # ax.elav=20
    ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
    ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
    ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
    navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    # navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    fig.suptitle(f"Cluster{cluster_n} OL R")
    fig.savefig(os.path.join(fig_save_path,f'Location_{selected_cluster_n}clusters_{current_data}_excludeMajor-{exclude_neurons}_cluster{cluster_n}_OL-R.pdf'))

#%%

# %%
