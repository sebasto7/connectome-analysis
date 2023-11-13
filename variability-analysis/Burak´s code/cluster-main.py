
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
from scipy.spatial.distance import hamming
from sklearn.metrics import pairwise_distances
from fafbseg import flywire
import navis

from sklearn.metrics.pairwise import cosine_similarity

#%% Load the data
main_path = '/Users/burakgur/Documents/Science/flywire-paper'
fig_save_path = os.path.join(main_path,"burak_figures")

current_data = 'Tm9_FAFB_L_R__relative.csv' 
# current_data = 'Tm9_abs_counts_L_5percent_threshold.csv' 

filePath =  os.path.join(main_path,'Tm9_rel_abs_counts',current_data)

data_df = pd.read_csv(filePath, header=0, index_col=0)
data_df = data_df.fillna(0)

# picked neurons = ['']
# cluster_df = data_df[['Mi4','CT1']]

# df_excluded = df.drop(columns=columns_to_exclude)

#TODO: Take binary data
# cluster_df = data_df[['Mi4','CT1']]
# neurons_to_exclude = ["L3", "Mi4", "CT1"]
# cluster_df = data_df.drop(columns=neurons_to_exclude)
cluster_df = data_df

data_array = cluster_df.to_numpy()
order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
         'C3','Tm1','PS125','L4','ML1','TmY17','C2',
         'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
data_df = data_df[order]
#%% K means clustering step 1, decide on the cluster N
data_array[np.isnan(data_array)] = 0

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_array)
scaled_features = data_array
# Select the appopriate number of clusters
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
    kmeans.fit(scaled_features)
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
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)
    
axs[1].plot(range(2, max_c), silhouette_coefficients)
axs[1].set_xticks(range(2, max_c))
axs[1].set_xlabel("Number of Clusters")
axs[1].set_ylabel("Silhouette Coefficient")
# fig.savefig(os.path.join(fig_save_path,f'Kmeans_clusters_onlyMi4CT1data_KmeansParams_.pdf'))

#%% Perform K means with desired cluster n
n_k_clusters = 2
kmeans = KMeans(
    init="random",
    n_clusters=n_k_clusters,
    n_init=10,
    max_iter=300,
    random_state=42)
kmeans.fit(scaled_features)
unique_clusters , counts = np.unique(kmeans.labels_, return_counts= True)
print(f'Neurons per cluster: {counts}' )

fig2, axs = plt.subplots(len(unique_clusters),figsize=(5, 10),sharey=True)
major_inputs_data = data_df[['L3','Mi4','CT1','Tm16','Dm12']]
for cluster in unique_clusters:
    sns.boxplot(data=major_inputs_data.iloc[np.where(kmeans.labels_ == cluster)[0]],ax = axs[cluster])
    # axs[cluster].set_ylabel('relative counts')
fig2.suptitle(f'Neurons per cluster: {counts}')

#%% Plot correlations in all the data
major_inputs_data = data_df[['L3','Mi4','CT1']]

fig3 = plt.figure(figsize=[20,20])

g = sns.PairGrid(major_inputs_data)
g.map_lower(sns.regplot,scatter_kws = {'alpha': 0.3,'s':5})
g.map_upper(sns.regplot,scatter_kws = {'alpha': 0.3,'s':5})

def annotate_corr(x, y, **kwargs):
    corr_value, p = pearsonr(x,y)

    corr_text = f"r: {corr_value:.2f}\np: {p:.5f}\n"
    ax = plt.gca()
    ax.annotate(corr_text, xy=(0.5, 0.5), xycoords=ax.transAxes, fontsize=12,
                ha="center", va="center", color="r")

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=22)  # Adjust fontsize here
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=22)  # Adjust fontsize here


g.map_upper(annotate_corr)
g.map_lower(annotate_corr)

# g.map_diag(sns.regplot,scatter_kws = {'alpha': 0.1,'s':3})
g.map_diag(sns.histplot)

#%% Correlation matrix
fig44 = plt.figure(figsize=[10,10])

order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
         'C3','Tm1','PS125','L4','ML1','TmY17','C2',
         'OA-AL2b2']
data_df = data_df[order]

correlation_matrix = data_df.corr()
# Set diagonals to NaN
for i in range(len(correlation_matrix)):
    correlation_matrix.iloc[i, i] = float('nan')
    

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)

# Calculate p-values for correlations (this is just an example calculation)
# Replace this with your actual method for calculating significance
# Calculate p-values for correlations
p_values = np.zeros((len(correlation_matrix), len(correlation_matrix)))
for i, col1 in enumerate(correlation_matrix.columns):
    for j, col2 in enumerate(correlation_matrix.columns):
        if i != j:
            corr, p_value = pearsonr(data_df[col1], data_df[col2])
            p_values[i, j] = p_value
dim = p_values.shape[0]
bonferonni_c =(dim*dim/2)-dim #bonferonni correction to adjust the alpha 
alpha1 = 0.05/bonferonni_c
alpha2 = 0.005/bonferonni_c
alpha3 = 0.001/bonferonni_c
# Add stars on significant correlations
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if p_values[i, j] < alpha3:
            plt.text(j + 0.5, i + 0.5, '***', ha='center', va='center', fontsize=10, color='red')
        elif p_values[i, j] < alpha2:
            plt.text(j + 0.5, i + 0.5, '**', ha='center', va='center', fontsize=10, color='red')
        elif p_values[i, j] < alpha1:  # You can adjust the significance threshold
            plt.text(j + 0.5, i + 0.5, '*', ha='center', va='center', fontsize=10, color='red')

# Show the plot
plt.show()

fig44.savefig(os.path.join(fig_save_path,f'Correlation_{current_data}_.pdf'))

#%%
from scipy.stats import linregress

# Generate example data
x = data_df['Mi4']
y = data_df['CT1']

# Calculate correlation coefficient and p-value
correlation_coefficient, p_value = np.corrcoef(x, y)[0, 1], linregress(x, y).pvalue

# Create a scatter plot of the data points
plt.scatter(x, y, label="Data Points")

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(x, y)
fit_line = slope * x + intercept
plt.plot(x, fit_line, color='red', label="Fit Line")

# Annotate correlation coefficient and p-value
annotation = f"Correlation: {correlation_coefficient:.2f}\nP-Value: {p_value:.4f}"
plt.annotate(annotation, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)

# Set labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.show()
#%% Hierarchical clustering based on counts
n_h_clusters = 6
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_h_clusters, affinity='euclidean', linkage='ward')
labels_hierarchical = hierarchical_cluster.fit_predict(scaled_features)

unique_clusters , counts = np.unique(labels_hierarchical, return_counts= True)

fig3, axs = plt.subplots(len(unique_clusters),figsize=(5, 10),sharey=True)
major_inputs_data = data_df[['L3','Mi4','CT1','Tm16','Dm12']]
for cluster in unique_clusters:
    sns.boxplot(data=major_inputs_data.iloc[np.where(labels_hierarchical == cluster)[0]],ax = axs[cluster])
    # axs[cluster].set_ylabel('relative counts')
fig3.suptitle(f'Neurons per cluster (hierarchical): {counts}')
#%% Dendogram hamming
# Calculate pairwise Hamming distances
binary_array = (data_array>0).astype(int)
distances = pairwise_distances(binary_array, metric="hamming")


# Perform hierarchical clustering using linkage
linkage_matrix = hierarchy.linkage(distances, method='ward')  # You can choose different linkage methods
# Reorder data based on dendrogram leaves
reordered_indices = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
reordered_array = binary_array[reordered_indices]
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
sns.heatmap(reordered_distances, cmap='coolwarm', annot=False, xticklabels=cluster_df.index, yticklabels=cluster_df.index, ax=ax_heatmap, cbar=False)
#ax_heatmap.set_title('Cosine Similarity Heatmap')
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
n_h_clusters = 6
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_h_clusters, affinity='euclidean', linkage='ward')
labels_hierarchical = hierarchical_cluster.fit_predict(binary_array)

unique_clusters , counts = np.unique(labels_hierarchical, return_counts= True)

fig3, axs = plt.subplots(len(unique_clusters),figsize=(5, 10),sharey=True)
major_inputs_data = data_df[['L3','Mi4','CT1','Tm16','Dm12']]
for cluster in unique_clusters:
    sns.boxplot(data=major_inputs_data.iloc[np.where(labels_hierarchical == cluster)[0]],ax = axs[cluster])
    # axs[cluster].set_ylabel('relative counts')
fig3.suptitle(f'Neurons per cluster (hierarchical): {counts}')


#%% Cosine similarity

# Calculate cosine similarity
cosine_sim = cosine_similarity(scaled_features)
# Convert the cosine_sim 2D array to a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=cluster_df.index, columns=cluster_df.index)

# Perform hierarchical clustering
dendrogram_cosine = hierarchy.linkage(cosine_sim, method='ward')
cosine_row_order = hierarchy.leaves_list(dendrogram_cosine)
# Create a new DataFrame with reordered rows and columns
cluster_df_reordered_cosine_sim = cluster_df.iloc[cosine_row_order].copy()
# Calculate cosine similarity
cosine_sim_reordered = cosine_similarity(cluster_df_reordered_cosine_sim.values)
# Convert the cosine_sim 2D array to a DataFrame
cosine_sim_reordered_df = pd.DataFrame(cosine_sim_reordered, index=cluster_df_reordered_cosine_sim.index, columns=cluster_df_reordered_cosine_sim.index)


#Data
_data = cluster_df_reordered_cosine_sim
_data.dropna(how='all', inplace=True)    # now dropping if all values in the row are nan

# Create a figure with custom grid layout
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[8, 1], height_ratios=[1.2, 8, 0.5])

# Plot the dendrogram_cosine
ax_dendrogram_cosine = plt.subplot(gs[0, :-1])
ax_dendrogram_cosine.spines['top'].set_visible(False)
ax_dendrogram_cosine.spines['right'].set_visible(False)
ax_dendrogram_cosine.spines['bottom'].set_visible(False)
ax_dendrogram_cosine.spines['left'].set_visible(False)
ax_dendrogram_cosine.get_xaxis().set_visible(False)
ax_dendrogram_cosine.get_yaxis().set_visible(False)
hierarchy.dendrogram(dendrogram_cosine, ax=ax_dendrogram_cosine, color_threshold=0)

# Plot the heatmap using the reordered DataFrame
ax_heatmap = plt.subplot(gs[1, :-1])
sns.heatmap(cosine_sim_reordered, cmap='coolwarm', annot=False, xticklabels=_data.index, yticklabels=_data.index, ax=ax_heatmap, cbar=False,center=0)
#ax_heatmap.set_title('Cosine Similarity Heatmap')
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
cbar.set_label('Cosine Similarity')

#%% PCA and plot some clusters if you want
#%% PCA
# NaN means the Tm9 neuron did not receive any input from that neuron
data_array = cluster_df.to_numpy(dtype=float,copy=True).T

# Standardize
# data_array_binary = np.copy(data_array)
# data_array_binary[data_array>0] = 1
data_array_norm = data_array-data_array.mean(axis=0)
data_array_norm /= data_array_norm.std(axis=0)
n = data_array_norm.shape[0]

# Cov matrix and eigenvectors
cov = (1/n) * data_array_norm @ data_array_norm.T
eigvals, eigvecs = np.linalg.eig(cov)
k = np.argsort(eigvals)[::-1]
eigvals = eigvals[k]
eigvecs = eigvecs[:,k]
# %% Clusters on PCs
fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array.T @ eigvecs[:,pc_1]
pc2 = data_array.T @ eigvecs[:,pc_2] 


plt.scatter(pc1[kmeans.labels_==0],pc2[kmeans.labels_==0],color='g',label='Cluster 1')
plt.scatter(pc1[kmeans.labels_==1],pc2[kmeans.labels_==1],color='b',label='Cluster 2')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
# %% Save figures
# fig1.savefig(os.path.join(fig_save_path,f'Kmeans_clusters_{n_k_clusters}_{current_data}_onlyMi4CT1_KmeansParams_.pdf'))
# fig2.savefig(os.path.join(fig_save_path,f'Kmeans_clusters_{n_k_clusters}_{current_data}_onlyMi4CT1_main_inputs_.pdf'))
# # g.savefig(os.path.join(fig_save_path,f'PairGrid_{current_data}_main_inputs_.pdf'))
# fig3.savefig(os.path.join(fig_save_path,f'Hierachical_clusters_{n_h_clusters}_{current_data}_onlyMi4CT1_main_inputs_.pdf'))


fig1.savefig(os.path.join(fig_save_path,f'Kmeans_clusters_{n_k_clusters}_{current_data}_allData_KmeansParams_.pdf'))
fig2.savefig(os.path.join(fig_save_path,f'Kmeans_clusters_{n_k_clusters}_{current_data}_allData_main_inputs_.pdf'))
# g.savefig(os.path.join(fig_save_path,f'PairGrid_{current_data}_main_inputs_.pdf'))
fig3.savefig(os.path.join(fig_save_path,f'Hierachical_clusters_{n_h_clusters}_{current_data}_allData_main_inputs_.pdf'))



#%% Visualize the clusters
filePath =  os.path.join(main_path,"Tm9 proofreadings - Hoja 1.csv")
location_df = pd.read_csv(filePath, header=0)

# %%
OL_labels = [string.split(":")[2] for string in data_df.index]
ids= location_df["optic_lobe_id"]


xyz = np.zeros([len(OL_labels),3])
for idx, neuron in enumerate(OL_labels):
    df_loc = np.where(ids==neuron)[0]
    coordinate = location_df.iloc[df_loc]["XYZ-ME"].to_numpy(dtype=str, copy=True)
    xyz[idx,:] = np.array([coordinate[0].split(',')],dtype=float)
xyz *=[4,4,40] # For plotting it using navis

# %%
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
ax.scatter(xyz[kmeans.labels_==0,0],xyz[kmeans.labels_==0,1],xyz[kmeans.labels_==0,2],'.',color=[0,1,0,0.4])
ax.scatter(xyz[kmeans.labels_==1,0],xyz[kmeans.labels_==1,1],xyz[kmeans.labels_==1,2],'.',color=[0,0,1,0.4])
OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=200
ax.elav=20
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '20 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '20 um')


 # %%
from matplotlib.colors import Normalize

# Set the center value for the colormap
center_value = 0

# Create a Normalize instance to set the colormap center
norm = Normalize(vmin=pc2.min(), vmax=pc2.max())
norm_centered = Normalize(vmin=pc2.min() - center_value, vmax=pc2.max() - center_value)


fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],'.',c=pc2 - center_value,cmap='viridis',norm=norm_centered)
# cbar = plt.colorbar()
# cbar.set_label('PC2 value')

OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=200
ax.elav=20
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '20 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '20 um')

# %%
