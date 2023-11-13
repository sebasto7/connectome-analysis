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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
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

#%% Perform DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=20)
labels = dbscan.fit_predict(data_array)

unique_clusters , counts = np.unique(labels, return_counts= True)
print(f'Neurons per cluster: {counts}' )

fig2, axs = plt.subplots(len(unique_clusters),figsize=(5, 10),sharey=True)
major_inputs_data = data_df[['L3','Mi4','CT1','Tm16','Dm12']]
for cluster in unique_clusters:
    sns.boxplot(data=major_inputs_data.iloc[np.where(labels == cluster)[0]],ax = axs[cluster])
    # axs[cluster].set_ylabel('relative counts')
fig2.suptitle(f'Neurons per cluster: {counts}')

# %% Save figures
# %% Save figures
fig1.savefig(os.path.join(fig_save_path,f'DBSCAN_Silhouette_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_scale-{scale_data}.pdf'))
fig2.savefig(os.path.join(fig_save_path,f'DBSCAN_neurons_{n_k_clusters}clusters_{current_data}_excludeMajor-{exclude_neurons}_scale-{scale_data}.pdf'))

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

# fig4, axs4 = plt.subplots(selected_cluster_n,figsize=(16, 16))
cluster_labels= labels
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
    fig.suptitle(f"Cluster{cluster_n} OL L")
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
