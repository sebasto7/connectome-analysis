#
# - Clustering of Tm9 synaptic connectivity data - 
#
#%% Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import matplotlib.gridspec as gridspec
from fafbseg import flywire
import navis


#%% Input parameters
scale_data = False

exclude_neurons = False
neurons_to_exclude = ["L3", "Mi4", "CT1"]

#%% Load the data
main_path = '/Users/burakgur/Documents/Science/flywire-paper'
fig_save_path = os.path.join(main_path,"burak_figures","k-means")

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
    
#%%# Select the appopriate number of clusters
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

fig1, axs = plt.subplots(2)
# A list holds the SSE values for each k
sse = []
max_c = 15
for k in range(1, max_c):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_array)
    sse.append(kmeans.inertia_)
    
axs[0].plot(range(1, max_c), sse)
axs[0].set_xticks(range(1, max_c))
axs[0].set_xlabel("Number of Clusters")
axs[0].set_ylabel("SSE")
   
# Silhu
silhouette_coefficients = []
   
# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, max_c):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_array)
    score = silhouette_score(data_array, kmeans.labels_)
    silhouette_coefficients.append(score)
    
axs[1].plot(range(2, max_c), silhouette_coefficients)
axs[1].set_xticks(range(2, max_c))
axs[1].set_xlabel("Number of Clusters")
axs[1].set_ylabel("Silhouette Coefficient")

#%% Perform K means with desired cluster n
n_k_clusters = 2
kmeans = KMeans(
    init="random",
    n_clusters=n_k_clusters,
    n_init=10,
    max_iter=300,
    random_state=42)
kmeans.fit(data_array)
unique_clusters , counts = np.unique(kmeans.labels_, return_counts= True)
print(f'Neurons per cluster: {counts}' )

fig2, axs = plt.subplots(len(unique_clusters),figsize=(5, 15),sharey=True)
# major_inputs_data = data_df[['L3','Mi4','CT1','Tm16','Dm12']]
# order
for cluster in unique_clusters:
    sns.boxplot(data=data_df.iloc[np.where(kmeans.labels_ == cluster)[0]],ax = axs[cluster])
    
    if cluster == len(unique_clusters)-1:
        axs[cluster].set_xticklabels(axs[cluster].get_xticklabels(),rotation=90)
    else:
        axs[cluster].set_xticks([])
    # axs[cluster].set_ylabel('relative counts')
fig2.suptitle(f'Neurons per cluster: {counts}')

# %% Save figures
fig1.savefig(os.path.join(fig_save_path,f'k-means_Silhouette_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_scale-{scale_data}.pdf'))
fig2.savefig(os.path.join(fig_save_path,f'k-means_neurons_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_scale-{scale_data}.pdf'))

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
cluster_labels= kmeans.labels_

#%% plot 2 clusters
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

# OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=200
ax.elav=40
# ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
# ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
# ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
# navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
ax.scatter(xyz[cluster_labels==0,0],xyz[cluster_labels==0,1],
        xyz[cluster_labels==0,2],'.',linewidth=0,s=15,color=[228/255,26/255,28/255,0.8], label=f"Cluster 0")
ax.scatter(xyz[cluster_labels==1,0],xyz[cluster_labels==1,1],
        xyz[cluster_labels==1,2],'.',linewidth=0,s=15,color=[55/255,126/255,184/255,0.8], label=f"Cluster 1")

fig.suptitle(f"Clusters in OL L")
fig.savefig(os.path.join(fig_save_path,f'Location_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_OL-L.pdf'))

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
# OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=340
ax.elav=20
ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
# navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
ax.scatter(xyz[cluster_labels==0,0],xyz[cluster_labels==0,1],
        xyz[cluster_labels==0,2],'.',linewidth=0,s=15,color=[228/255,26/255,28/255,0.8], label=f"Cluster 0")
ax.scatter(xyz[cluster_labels==1,0],xyz[cluster_labels==1,1],
        xyz[cluster_labels==1,2],'.',linewidth=0,s=15,color=[55/255,126/255,184/255,0.8], label=f"Cluster 1")
fig.suptitle(f"Clusters in OL R")
fig.savefig(os.path.join(fig_save_path,f'Location_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_OL-R.pdf'))

#%% plot pc


#%%
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
    navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    # fig.suptitle(f"Cluster{cluster_n} OL L")
    fig.savefig(os.path.join(fig_save_path,f'Location_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_cluster{cluster_n}_OL-L.pdf'))

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
    navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    fig.suptitle(f"Cluster{cluster_n} OL R")
    fig.savefig(os.path.join(fig_save_path,f'Location_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_cluster{cluster_n}_OL-R.pdf'))


# %%
