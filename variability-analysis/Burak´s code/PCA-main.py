#%% Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from fafbseg import flywire
import navis

#%% Load the data
main_path = '/Users/burakgur/Documents/Science/flywire-paper'
fig_save_path = os.path.join(main_path,"burak_figures")

current_data = 'Tm9_FAFB_L_R__relative.csv' 
filePath =  os.path.join(main_path,'Tm9_rel_abs_counts',current_data)

data_df = pd.read_csv(filePath, header=0, index_col=0)
order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
         'C3','Tm1','PS125','L4','ML1','TmY17','C2',
         'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
data_df = data_df[order]

#%% PCA with sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming X is your data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.to_numpy(dtype=float,copy=True))


# Create a PCA instance
n_components = 20  # You can adjust this according to your needs
pca = PCA(n_components=n_components)

# Fit PCA to the scaled data
pca.fit(X_scaled)
eigenvectors = pca.components_

X_pca = pca.transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Result')
plt.show()
#%%
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance ratio
plt.bar(range(1, n_components + 1), explained_variance_ratio, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.show()

#%% PCA
# NaN means the Tm9 neuron did not receive any input from that neuron
data= data_df.fillna(0) # replace NaN with 0s
data_array = data.to_numpy(dtype=float,copy=True)

# Standardize
# data_array_binary = np.copy(data_array)
# data_array_binary[data_array>0] = 1
# Features are in rows now (due to transpose)
data_array_norm = data_array-data_array.mean(axis=0)
data_array_norm /= data_array_norm.std(axis=0)
n = data_array_norm.shape[0]
# data_array_norm = data_array.copy()

# Cov matrix and eigenvectors
cov = (1/n) * data_array_norm.T @ data_array_norm
eigvals, eigvecs = np.linalg.eig(cov)
k = np.argsort(eigvals)[::-1]
eigvals = eigvals[k]
eigvecs = eigvecs[:,k]
#%% Explained variance of PCs
#plot the square-root eigenvalue spectrum
fig = plt.figure(figsize=[5,5])
explained_var = np.cumsum(eigvals)/max(np.cumsum(eigvals))*100
explained_var = np.roll(explained_var,1)
explained_var[0] = 0
explained_var= np.append(explained_var,(np.cumsum(eigvals)/max(np.cumsum(eigvals))*100)[-1])
plt.plot(explained_var,'-o',color='black')
plt.xlabel('dimensions')
plt.ylabel('explained var (percentage)')
plt.xlim([0,20])
plt.ylim([0,100])
plt.xticks(range(21))
# plt.title('Explained var (percentage)')
plt.title(f'explained variances {np.around(explained_var[0:3],2)}...')
plt.show()
fig.savefig(os.path.join(fig_save_path,'PCA_varExplained_relative.pdf'))

#%%
fig = plt.figure()
pc_1 = 0
pc_2 = 1
plt.scatter(data_array_norm @ eigvecs[:,pc_1],data_array_norm @ eigvecs[:,pc_2])
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_data_relative.pdf'))

# %% Contributions
fig = plt.figure(figsize=[4,6])
plt.imshow(np.array([eigvecs[:,0],eigvecs[:,1]]).T,cmap='coolwarm',aspect='auto')
# plt.imshow(np.array([eigvecs[:4,0],eigvecs[:4,1],eigvecs[:4,2],eigvecs[:4,3]]).T,cmap='coolwarm',aspect='auto')
plt.colorbar()
plt.xlabel('Principal components (PCs)')
ax = plt.gca()
a = list(range(0, eigvecs.shape[0]))
ax.set_yticks(a)
ax.set_yticklabels(data.columns)
plt.title('Contribution of neurons to PCs')
fig.savefig(os.path.join(fig_save_path,'PCA_PC_contributions_relative.pdf'))
# %% Dorso ventral differences in PCA?
delimiter = ":"

# Extract letters after the delimiter for each string
dv_labels = [string.split(delimiter)[3] if delimiter in string else "" for string in data_df.index]

fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array_norm @ eigvecs[:,pc_1]
pc2 = data_array_norm @ eigvecs[:,pc_2] 

d_labels = ["D" in string for string in dv_labels]
v_labels = ["V" in string for string in dv_labels]
plt.scatter(pc1[d_labels],pc2[d_labels],color=[152/255,78/255,163/255,0.8],label='dorsal')
plt.scatter(pc1[v_labels],pc2[v_labels],color=[77/255,175/255,74/255,0.8],label='ventral')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_DV.pdf'))
# %% R and Left
delimiter = ":"

# Extract letters after the delimiter for each string
rl_labels = [string.split(delimiter)[2][0] if delimiter in string else "" for string in data_df.index]

fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array_norm @ eigvecs[:,pc_1]
pc2 = data_array_norm @ eigvecs[:,pc_2] 

r_labels = ["R" in string for string in rl_labels]
l_labels = ["L" in string for string in rl_labels]
plt.scatter(pc1[r_labels],pc2[r_labels],color='r',label='right')
plt.scatter(pc1[l_labels],pc2[l_labels],color='g',label='left')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_RL.pdf'))
# %% K means
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
# %% Plot clujsters

# Extract letters after the delimiter for each string

fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array_norm @ eigvecs[:,pc_1]
pc2 = data_array_norm @ eigvecs[:,pc_2] 

plt.scatter(pc1[kmeans.labels_==0],pc2[kmeans.labels_==0],color=[228/255,26/255,28/255,0.8],label='Cluster 1')
plt.scatter(pc1[kmeans.labels_==1],pc2[kmeans.labels_==1],color=[55/255,126/255,184/255,0.8],label='Cluster 2')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_KmeansClusters.pdf'))

#%% Visualize the PC on the mesh
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
#%%
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')


OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']
ax.azim=340
ax.elav=20
# ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
# ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
# ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],'.',linewidth=0,s=15,c = pc2,cmap='copper',vmin=-10,vmax=10)

fig.suptitle(f"PC2 in OL L")
fig.savefig(os.path.join(fig_save_path,f'Location_PC2_{current_data}_OL-R.pdf'))

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=0
ax.elav=20
# ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
# ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
# ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],'.',linewidth=0,s=15,c = pc2,cmap='copper',vmin=-10,vmax=10)

fig.suptitle(f"PC2 in OL R")
fig.savefig(os.path.join(fig_save_path,f'Location_PC2_{current_data}_OL-L.pdf'))

3# %%

# %%Dorsoventral visualization
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')


# OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']
ax.azim=200
ax.elav=20
# ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
# ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
# ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
# navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')

ax.scatter(xyz[d_labels,0],xyz[d_labels,1],xyz[d_labels,2],'.',linewidth=0,s=15,color=[152/255,78/255,163/255],label='dorsal')
ax.scatter(xyz[v_labels,0],xyz[v_labels,1],xyz[v_labels,2],'.',linewidth=0,s=15,color=[77/255,175/255,74/255],label = 'ventral')

fig.suptitle(f"DV in OL L")
fig.savefig(os.path.join(fig_save_path,f'Location_DV_{current_data}_OL-R.pdf'))

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
OL_R = flywire.get_neuropil_volumes(['ME_R']) #['ME_R','LO_R','LOP_R']
OL_L = flywire.get_neuropil_volumes(['ME_L']) #['ME_R','LO_R','LOP_R']

ax.azim=340
ax.elav=20
# ax.set_xlim(min(xyz[:,0])-10, max(xyz[:,0])-300000)
# ax.set_ylim(min(xyz[:,1])-10, max(xyz[:,1])-100000)
# ax.set_zlim(min(xyz[:,2])-10, max(xyz[:,2])-100)
navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
# navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
ax.scatter(xyz[d_labels,0],xyz[d_labels,1],xyz[d_labels,2],'.',linewidth=0,s=15,color=[152/255,78/255,163/255],label='dorsal')
ax.scatter(xyz[v_labels,0],xyz[v_labels,1],xyz[v_labels,2],'.',linewidth=0,s=15,color=[77/255,175/255,74/255],label = 'ventral')

fig.suptitle(f"DV in OL R")
fig.savefig(os.path.join(fig_save_path,f'Location_DV_{current_data}_OL-L.pdf'))

# %%
