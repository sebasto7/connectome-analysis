# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:34:16 2022

@author: smolina

Test-Navi
"""

#%% Importing packages

import navis
import fafbseg
from fafbseg import flywire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import _pickle as cPickle # For Python 3.X
import os

#Setting a FlyWire secret (do not run this everytime)
#fafbseg.flywire.set_chunkedgraph_secret("5719b2db462d94d6aa0e903c1ff889e4")

# Saving options
save_excel_file = False
save_meshes = False

outDir = r'D:\FlyWire\Meshes'
baseName = 'Tm9s'
extension='.pickle'


#%% Getting data from FlyWire


# Check if root IDs are outdated (i.e. have more recent edits)
r1 = 720575940609401976 # Mi4
r2 = 720575940627821903 # Tm9
r3 = 720575940624797733 # C3
r4 = 720575940629721687 # L3
r5 = 720575940618885205 # Tm9
flywire.is_latest_root([r5,r2])

# Selecting Neurons
m1 = flywire.get_mesh_neuron(r1) #Mi4
m2 = flywire.get_mesh_neuron(r2) #Tm9
m3 = flywire.get_mesh_neuron(r3) #C3
m4 = flywire.get_mesh_neuron(r4) #L3
m5= flywire.get_mesh_neuron(r5) #Tm9

#Selecting more neurons together
root_ids = np.array([r2,r5], dtype=np.int64)
m_all = flywire.get_mesh_neuron(root_ids)


#%% Select an specific neuron

neuron = m5
# Fetch the neuron's inputs
inputs = flywire.synapses.fetch_synapses(neuron, pre=False, post=True, attach=True, 
                                         min_score=30, clean=True, transmitters=False, 
                                         neuropils=False, live_query=True, batch_size=30, 
                                         dataset='production', progress=True)
#synaptic_counts = flywire.synapses.synapse_counts(root_ids, by_neuropil=False, min_score=30, live_query=True,batch_size=10, dataset='production')



#%% Counting inputs per ID
inputs_count = {}
inputs_str = inputs.applymap(str)
for c in inputs_str['pre'].to_list():
    inputs_count[c] = inputs_count.get(c, 0) + 1
input_count_df = pd.DataFrame(inputs_count, index=[0])
input_count_df = input_count_df.T
input_count_df.rename(columns={0: "counts"},inplace=True)
input_count_df.index.names = ['presynaptic_ID']



#%% Saving options
# Excel files
if save_excel_file:
    inputs_str = inputs.applymap(str)
    import xlsxwriter
    file_name = str(neuron.id)+'_inputs.xlsx'
    inputs.to_excel(file_name)
    file_name = 'U:\\' + str(neuron.id)+'_inputs.xlsx'
    inputs_str.to_excel(file_name)
    file_name = 'U:\\' + str(neuron.id)+'_inputs_count.xlsx'
    input_count_df.to_excel(file_name)
# pickle or feather files
if save_meshes:
    savePath = os.path.join(outDir, baseName + extension)
    saveVar = open(savePath, "wb")
    cPickle.dump(m_all, saveVar, protocol=-1)
    saveVar.close()
    print('Seb, codding here, ~line 96')
    pass



#%%
##################### Plotting Inputs in 3D ####################

# Get the x/y/z coordinates as (N, 3) array
xyz = inputs[['pre_x', 'pre_y', 'pre_z']].values
#input_ids = inputs['id', 'valid']

# Passing (N, 3) array to plot3d will produce a scatterplot
# (see `scatter_kws` argument for customization)
fig = navis.plot3d([neuron, xyz], backend='auto', color = 'blue', scatter_kws ={"color":'red', "size":2},
                   clear = True)


#%%
##################### Plotting many neurons in 3D ####################

# Fetch the neuron's inputs
neuron = m2
presypnaptic_neuron= m4
color_dict = {neuron.id:'blue',presypnaptic_neuron.id:'red' }
inputs = flywire.synapses.fetch_synapses(neuron, pre=False, post=True, attach=True, 
                                         min_score=30, clean=True, transmitters=False, 
                                         neuropils=False, live_query=True, batch_size=30, 
                                         dataset='production', progress=True)

# Get the x/y/z coordinates as (N, 3) array
xyz_all = inputs[['pre_x', 'pre_y', 'pre_z']].values
is_presypnaptic =  inputs['pre']==m4.id
xyz_pre = inputs[is_presypnaptic][['pre_x', 'pre_y', 'pre_z']].values

# Passing (N, 3) array to plot3d will produce a scatterplot
# (see `scatter_kws` argument for customization)
OL_R = flywire.get_neuropil_volumes(['ME_R','LO_R','LOP_R']) # ['LOP_L', 'LOP_R'], ['LO_R'], ['ME_R']
fig = navis.plot3d([m_all, xyz_pre], backend='auto', 
                   color = color_dict, 
                   scatter_kws ={"color":'green', "size":8},
                   clear = True)
navis.plot2d([m_all,OL_R], method='3d_complex',color=color_dict) #scalebar = '20 um'


#%%
########################## Plotting many neurons in 2D ###############################
sk_m2 = flywire.skeletonize_neuron(r2, progress=False)
sk_m4 = flywire.skeletonize_neuron(r4, progress=False)
sk_m_all = flywire.skeletonize_neuron(root_ids)
neuron = sk_m2
presypnaptic_neuron= sk_m4
color_dict = {neuron.id:'blue',presypnaptic_neuron.id:'red' }
inputs = flywire.synapses.fetch_synapses(neuron, pre=False, post=True, attach=True, 
                                         min_score=30, clean=True, transmitters=True, 
                                         neuropils=False, live_query=True, batch_size=30, 
                                         dataset='production', progress=True)

# Get the x/y/z coordinates as (N, 3) array
xyz_all = inputs[['pre_x', 'pre_y', 'pre_z']]
is_presypnaptic =  inputs['pre']==m4.id # Taking only inputs from m4
xyz_pre = inputs[is_presypnaptic][['pre_x', 'pre_y', 'pre_z']]

OL_R = flywire.get_neuropil_volumes(['ME_R','LO_R','LOP_R']) # ['LOP_L', 'LOP_R'], ['LO_R'], ['ME_R']
fig, ax = navis.plot2d([sk_m_all,OL_R],linewidth=2, linestyle='--', method='2d',color=color_dict) #scalebar = '20 um'
ax.scatter(xyz_pre['pre_x'], xyz_pre['pre_y'], c='green', marker='o', s= 5)
plt.savefig("U:\L3_Tm9.pdf", format="pdf", bbox_inches="tight")
plt.show() 

fig,ax =  navis.plot2d([m_all,OL_R], method='2d',color=color_dict) #scalebar = '20 um'
ax.scatter(xyz_pre['pre_x'], xyz_pre['pre_y'], c='green', marker='o', s= 2)
plt.savefig("U:\L3_Tm9.pdf", format="pdf", bbox_inches="tight")
plt.show() 

# Plotting neurotranmitter prediction
outputs = flywire.fetch_synapses(r4, pre=True, post=False, live_query=True,transmitters=True)
is_postsypnaptic =  outputs['post']==m2.id # Taking only outputs to m2
fig, ax = fafbseg.synapses.plot_nt_predictions(outputs)
plt.savefig("U:\prediction_NT.pdf", format="pdf", bbox_inches="tight")
plt.show()

#%%
############# Plotting Inputs and Outputs in 3 D (using vispy) ################

# Fetch the inputs and attach them as .connectors property
# See `neuron.connectors` 
neuron = m2
_ = flywire.synapses.fetch_synapses(m2, pre=True, post=True, attach=True,
                                    min_score=30, clean=True, transmitters=False, 
                                    neuropils=False, live_query=True, batch_size=30, 
                                    dataset='production', progress=True)

# Plot neuron plus its connectors
# (there are various ways to customize how connectors are plotted)
fig1 = navis.plot3d(neuron, connectors=True, backend='auto', color = 'blue', 
                    cn_colors = {"post": "red", "pre": "green"},scatter_kws ={"size":2},
                    clear = True)


