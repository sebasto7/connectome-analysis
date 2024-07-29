# -*- coding: utf-8 -*-
"""
Created on Mon July 22 11:00:00 2024

@author: smolina

optic flow analysis of DS cells in FAFB data set
"""

#%%


## Importing some packages
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns
import os
from fafbseg import flywire
from scipy.stats import gaussian_kde

## Importing custom fnuctions

from optic_flow_analysis_helper import plot_hex_grid, draw_vector, calculate_new_p_values,add_space_in_between

#%%

# Plotting settings

save_figures = True
save_path = r'D:\Connectomics-Data\FlyWire\Pdf-plots'

color_dict = {'T4a':'g','T4b':'b', 'T4c':'r', 'T4d':'y'}


#%%

## General settings

cell_for_grid = 'Mi1'
to_cell_of_interest_ls = ['T4a','T4b','T4c','T4d']
correlator = 'HR' # 'BL' for Barlow-Levick, 'HR' for Hassenstein-Reichert, 'HR-BL' for the combination

#%%



## Importing data from Codex
dataPath = R'D:\Connectomics-Data\FlyWire\Codex-datasets'
fileName = 'ol_columns.csv.gz'
filePath = os.path.join(dataPath,fileName)
columns_id_data = pd.read_csv(filePath, compression='gzip')
columns_id_data.rename(columns={"cell id": "root_id","cell type": "cell_type","column id": "column_id" }, inplace= True)

dataPath = R'D:\Connectomics-Data\FlyWire\Codex-datasets'
fileName = 'ol_metadata.csv.gz'
filePath = os.path.join(dataPath,fileName)
metadata = pd.read_csv(filePath, compression='gzip')

dataPath = R'D:\Connectomics-Data\FlyWire\Codex-datasets'
fileName = 'ol_connections.csv.gz'
filePath = os.path.join(dataPath,fileName)
connections = pd.read_csv(filePath, compression='gzip')
connections.rename(columns={"from cell id": "from_cell_id","to cell id": "to_cell_id" }, inplace= True)

dataPath = R'D:\Connectomics-Data\FlyWire\Codex-datasets'
fileName = 'column_assignment.csv.gz'
filePath = os.path.join(dataPath,fileName)
columns_id_coordinates = pd.read_csv(filePath, compression='gzip')




## Completing the "connections" data frame
connections_extended = connections.copy()

# Merge to add from_cell_type
connections_extended = connections_extended.merge(
    columns_id_data[['root_id', 'cell_type','column_id']],
    how='left',
    left_on='from_cell_id',
    right_on='root_id'
).rename(columns={'cell_type': 'from_cell_type','column_id': 'from_column_id'}).drop(columns=['root_id'])

# Merge to add to_cell_type
connections_extended = connections_extended.merge(
    columns_id_data[['root_id', 'cell_type','column_id']],
    how='left',
    left_on='to_cell_id',
    right_on='root_id'
).rename(columns={'cell_type': 'to_cell_type','column_id': 'to_column_id'}).drop(columns=['root_id'])


#%%

## Defining the grid or lattice

# Looking deeply at the match between individual cells and individual columns
cells_columns = columns_id_data[columns_id_data['cell_type'] == cell_for_grid]
number_unique_cells = len(cells_columns['root_id'].unique())
number_unique_columns = len(cells_columns['column_id'].unique())-1 #-1 to compensate for "not assigned label"

print(f'Total number of unique cells: {number_unique_cells}')
print(f'Total number of unique columns: {number_unique_columns}')
print(f'Missmatch: {number_unique_cells - number_unique_columns}')

# Identify duplicate values in 'column id'
duplicates = cells_columns[cells_columns.duplicated('column_id', keep=False)]

print(f'Number of duplicates / cells not assigned to columns: {len(duplicates)}')

# Filter out rows with repetitive values
filtered_cells_columns = cells_columns[~cells_columns['column_id'].isin(duplicates['column_id'])]
print(f'Filtered data length: {len(filtered_cells_columns)}')

# Addining column coordinates information
df_grid =  pd.merge(filtered_cells_columns, columns_id_coordinates[['root_id', 'p', 'q', 'x', 'y']], on='root_id', how='left')

#%%

## Data pre-analysis

'''
The ommatidia directions are well described by a hexagonal grid that we then aligned to the medulla column grid using 
the equator (+h) and central meridian (+v) as global landmarks
'''
h_x_ls = list(range(-8,8+1,1))
h_y_ls = list(range(8,-8-1,-1))
h = list(zip(h_x_ls,h_y_ls))

v_x_ls = list(range(-14,15+1,1))
v_y_ls = v_x_ls
v = list(zip(v_x_ls,v_y_ls))

p_y_ls = list(range(14,-13-1,-1))
p_x_ls = [0]* len(p_y_ls)
p = list(zip(p_x_ls,p_y_ls))

q_x_ls = list(range(-14,14+1,+1))
q_y_ls = [0]* len(q_x_ls)
q = list(zip(q_x_ls,q_y_ls))

# Given coordinates and labels (data)
# Original coordinates
original_p = df_grid.p.tolist()
original_q = df_grid.q.tolist()

# Shifting and spacing x coordinates
_relative_change = 0.58
_space = 1.15
spaced_original_p  = add_space_in_between(18, 18, _space, original_p)
new_p_values = calculate_new_p_values(spaced_original_p, original_q, start_key=-16, end_key=17, relative_change=_relative_change, in_space=0)
df_grid['new_p'] = new_p_values
center_shift = df_grid[df_grid.column_id == '628'].new_p.values[0] # Centering the original (0,0) coordinate again to 0,0
new_centered_p_values = new_p_values  - center_shift
df_grid['new_centered_p'] = new_centered_p_values



# "manual_labels" column ids
labels = df_grid.column_id.tolist() 
labels = None

#%%
## Ploting data on the regular grid in a loop


# Initializiny variables before the loop
angles_df_dict = {}

for cell in to_cell_of_interest_ls:

    to_cell_of_interest = cell

    # Getting inputs of the cell of interest
    BL_cell_of_interest = 'Mi4'

    cell_of_interest_inputs = connections_extended[(connections_extended.to_cell_type == to_cell_of_interest) & (connections_extended.from_cell_type == BL_cell_of_interest)].copy()

    # Sort by to_cell_id and synapses in descending order
    cell_of_interest_inputs_sorted = cell_of_interest_inputs.sort_values(by=['to_cell_id', 'synapses'], ascending=[True, False])

    # Group by to_cell_id and take the first row for each group (highest synapse value)
    BL_unique_highest_inputs = cell_of_interest_inputs_sorted.drop_duplicates(subset='to_cell_id', keep='first').copy()

    #Dropping 'not assigned' rows
    BL_unique_highest_inputs_filtered = BL_unique_highest_inputs[BL_unique_highest_inputs.to_column_id != 'not assigned'].copy()

    # Find the duplicates with the same to_cell_id and synapse value
    BL_duplicates_in_highest_inputs = cell_of_interest_inputs_sorted[cell_of_interest_inputs_sorted.duplicated(subset=['to_cell_id', 'synapses'], keep=False)]

    # Changing column names for clarity
    BL_unique_highest_inputs_filtered.rename(columns={'from_cell_id': 'BL_cell_id', 'from_cell_type': 'BL_cell_type', 'from_column_id': 'BL_column_id', 
                                                    'to_cell_id': 'home_cell_id', 'to_cell_type': 'home_cell_type','to_column_id': 'home_column_id'}, inplace = True)
    # Comvertions to strings
    BL_unique_highest_inputs_filtered['BL_cell_id'] = BL_unique_highest_inputs_filtered['BL_cell_id'].apply(str)


    # Getting inputs of the cell of interest
    HR_cell_of_interest = 'Mi9'

    cell_of_interest_inputs = connections_extended[(connections_extended.to_cell_type == to_cell_of_interest) & (connections_extended.from_cell_type == HR_cell_of_interest)].copy()

    # Sort by to_cell_id and synapses in descending order
    cell_of_interest_inputs_sorted = cell_of_interest_inputs.sort_values(by=['to_cell_id', 'synapses'], ascending=[True, False])

    # Group by to_cell_id and take the first row for each group (highest synapse value)
    HR_unique_highest_inputs = cell_of_interest_inputs_sorted.drop_duplicates(subset='to_cell_id', keep='first')

    #Dropping 'not assigned' rows
    HR_unique_highest_inputs_filtered = HR_unique_highest_inputs[HR_unique_highest_inputs.to_column_id != 'not assigned'].copy()

    # Find the duplicates with the same to_cell_id and synapse value
    HR_duplicates_in_highest_inputs = cell_of_interest_inputs_sorted[cell_of_interest_inputs_sorted.duplicated(subset=['to_cell_id', 'synapses'], keep=False)]

    # Changing column names for clarity
    HR_unique_highest_inputs_filtered.rename(columns={'from_cell_id': 'HR_cell_id', 'from_cell_type': 'HR_cell_type', 'from_column_id': 'HR_column_id', 
                                                    'to_cell_id': 'home_cell_id', 'to_cell_type': 'home_cell_type','to_column_id': 'home_column_id'}, inplace = True)
    # Comvertions to strings
    HR_unique_highest_inputs_filtered['HR_cell_id'] = HR_unique_highest_inputs_filtered['HR_cell_id'].apply(str)


    ## Combining the two correlators

    # Merge data frames of both correlators (with inner join to keep only the rows with keys that are present in both DataFrames)
    HR_BL_unique_highest_inputs_filtered = BL_unique_highest_inputs_filtered.merge(
        HR_unique_highest_inputs_filtered[['HR_cell_id', 'HR_cell_type','HR_column_id','home_cell_id']],
        how='inner',
        left_on='home_cell_id',
        right_on='home_cell_id'
    )

    #Dropping 'not assigned' rows
    HR_BL_unique_highest_inputs_filtered = HR_BL_unique_highest_inputs_filtered[HR_BL_unique_highest_inputs_filtered.BL_column_id != 'not assigned'].copy()
    HR_BL_unique_highest_inputs_filtered = HR_BL_unique_highest_inputs_filtered[HR_BL_unique_highest_inputs_filtered.HR_column_id != 'not assigned'].copy()


    ## Plotting on the 2D lattice: check access to each point and axis
    fig, ax, hexagons = plot_hex_grid(original_p, original_q, hex_size=0.5, spacing=1.0, fig_size=(20, 20), labels=labels, label_type='manual_labels', text_size=6)

    # Example to color hexagons based on a condition
    # for p_x, p_y in p:
    #     color_in_p = p_x
    #     color_in_q = p_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightgreen')

    # for q_x, q_y in q:
    #     color_in_p = q_x
    #     color_in_q = q_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightblue')
                
    
    for h_x, h_y in h:
        color_in_p = h_x
        color_in_q = h_y
        for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
            if x_pos == color_in_p and y_pos == color_in_q:
                hexagon.set_facecolor('lightyellow')


    # for v_x, v_y in v:
    #     color_in_p = v_x
    #     color_in_q = v_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightgrey')
                

    '''
    Ideally the nomenclature must relate: "start_ids"  with "from_column_id" and "end_ids" with "to_column_id"
    as written below. However, due to a logic mistake, in a previous step, I run what is written below this comment.
    I will need to fix that later. The current fast fix is correct in the analysis.

    start_ids = [int(x) for x in unique_highest_inputs_filtered.from_column_id.tolist()]
    end_ids = [int(x) for x in unique_highest_inputs_filtered.to_column_id.tolist()]

    '''

    #Preparing the vectors to draw
    if correlator == 'BL':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()] 
        from_cell_of_interest = BL_cell_of_interest
    elif correlator == 'HR':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()]
        from_cell_of_interest = HR_cell_of_interest
    elif correlator == 'HR-BL':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()]  
        from_cell_of_interest = [HR_cell_of_interest,BL_cell_of_interest]

    start_coords_ls = []
    end_coords_ls = []
    for start_id, end_id in zip(start_ids, end_ids):
        start_id_p = df_grid[df_grid.column_id == str(start_id)].p.values[0]
        start_id_q = df_grid[df_grid.column_id == str(start_id)].q.values[0]
        start_coords_ls.append((start_id_p,start_id_q))

        
        end_id_p = df_grid[df_grid.column_id == str(end_id)].p.values[0]
        end_id_q = df_grid[df_grid.column_id == str(end_id)].q.values[0]
        end_coords_ls.append((end_id_p,end_id_q))

    #Drawing vectors
    for start_coord, end_coord in zip(start_coords_ls,end_coords_ls):
        # Find the coordinates of the start and end hexagons
        x_start = start_coord[0]
        y_start = start_coord[1]
        
        x_end = end_coord[0]
        y_end = end_coord[1]
        
        draw_vector(ax, x_start, y_start, x_end, y_end, color=color_dict[to_cell_of_interest])


    plt.title(f'{correlator} correlator, {to_cell_of_interest}')
    #plt.show()

    # Saving plot
    if save_figures:
        figure_title = f'\Regular_grid_plot_vectors_{from_cell_of_interest}-home-{to_cell_of_interest}_correlator-{correlator}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization vectors on the 2D grid plotted')
    plt.close(fig)




    ## Quantifying angles realtuve to the qeuatorial axis (h)

    # Define the angle of the diagonal line y = -x (this is true h)
    # The angle of y = -x is -45 degrees (or -π/4 radians)
    diagonal_angle_rad = np.radians(-45)

    # Prepare the list for angles
    angles = []

    # Calculate the angle for each vector
    for start_id, end_id, start_coord, end_coord in zip(start_ids, end_ids, start_coords_ls, end_coords_ls):
        # Coordinates
        x_start, y_start = start_coord
        x_end, y_end = end_coord
        
        # Calculate the change in coordinates
        dx = x_end - x_start
        dy = y_end - y_start
        
        # Calculate the angle in radians relative to the x-axis
        angle_rad = np.arctan2(dy, dx)
        
        # Adjust the angle relative to the diagonal line y = -x
        adjusted_angle_rad = angle_rad - diagonal_angle_rad
        
        # Convert to degrees
        adjusted_angle_deg = np.degrees(adjusted_angle_rad)
        
        # Normalize the angle to be in the range [0, 360) degrees
        adjusted_angle_deg = (adjusted_angle_deg + 360) % 360
        
        # Append the result to the angles list
        angles.append((start_id, end_id, adjusted_angle_deg))

    # Create a DataFrame with the results
    angles_df = pd.DataFrame(angles, columns=['start_id', 'end_id', 'angle'])
    angles_df_dict[to_cell_of_interest] = angles_df





#%% Looking at distributions in the regular grid. Plotting in a loop
# Plotting
# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for cell in to_cell_of_interest_ls:
    # Data
    to_cell_of_interest = cell
    angles = angles_df_dict[to_cell_of_interest]['angle'] 
    angles_rad = np.radians(angles)  # Convert angles to radians  

    # Normalize angles to the range -π to π
    angles_rad_norm = np.mod(angles_rad + np.pi, 2 * np.pi) - np.pi  

    # KDE Plot
    sns.kdeplot(angles_rad_norm, ax=axs[0], color=color_dict[to_cell_of_interest], label=to_cell_of_interest)

    # Polar Plot
    axs[1] = plt.subplot(122, projection='polar')
    axs[1].hist(angles_rad, bins=30, density=False, edgecolor='black', color=color_dict[to_cell_of_interest], alpha=0.7, histtype='stepfilled', label=to_cell_of_interest)

# Set titles and labels outside the loop
axs[0].set_title(f'KDE of Angles \n{correlator} correlator')
axs[0].set_xlabel('Angle (radians)')
axs[0].set_ylabel('Density')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
axs[0].set_xticks(ticks)
axs[0].set_xticklabels(tick_labels)
axs[0].legend()

axs[1].set_title('Polar Plot of Angles')
axs[1].legend()

plt.tight_layout()
plt.suptitle(f'{correlator} correlator, {", ".join(to_cell_of_interest_ls)}', y=1.02)
#plt.show()

# Saving plot
if save_figures:
    figure_title = f'\Vectors_histogram_{from_cell_of_interest}-home_T4_correlator-{correlator}_from_regular_grid.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of vectors angles plotted')
plt.close(fig)


#%%
## Ploting data on the honeycomb-like grid in a loop


# Initializiny variables before the loop
angles_df_dict = {}

for cell in to_cell_of_interest_ls:

    to_cell_of_interest = cell

    # Getting inputs of the cell of interest
    BL_cell_of_interest = 'Mi4'

    cell_of_interest_inputs = connections_extended[(connections_extended.to_cell_type == to_cell_of_interest) & (connections_extended.from_cell_type == BL_cell_of_interest)].copy()

    # Sort by to_cell_id and synapses in descending order
    cell_of_interest_inputs_sorted = cell_of_interest_inputs.sort_values(by=['to_cell_id', 'synapses'], ascending=[True, False])

    # Group by to_cell_id and take the first row for each group (highest synapse value)
    BL_unique_highest_inputs = cell_of_interest_inputs_sorted.drop_duplicates(subset='to_cell_id', keep='first').copy()

    #Dropping 'not assigned' rows
    BL_unique_highest_inputs_filtered = BL_unique_highest_inputs[BL_unique_highest_inputs.to_column_id != 'not assigned'].copy()

    # Find the duplicates with the same to_cell_id and synapse value
    BL_duplicates_in_highest_inputs = cell_of_interest_inputs_sorted[cell_of_interest_inputs_sorted.duplicated(subset=['to_cell_id', 'synapses'], keep=False)]

    # Changing column names for clarity
    BL_unique_highest_inputs_filtered.rename(columns={'from_cell_id': 'BL_cell_id', 'from_cell_type': 'BL_cell_type', 'from_column_id': 'BL_column_id', 
                                                    'to_cell_id': 'home_cell_id', 'to_cell_type': 'home_cell_type','to_column_id': 'home_column_id'}, inplace = True)
    # Comvertions to strings
    BL_unique_highest_inputs_filtered['BL_cell_id'] = BL_unique_highest_inputs_filtered['BL_cell_id'].apply(str)


    # Getting inputs of the cell of interest
    HR_cell_of_interest = 'Mi9'

    cell_of_interest_inputs = connections_extended[(connections_extended.to_cell_type == to_cell_of_interest) & (connections_extended.from_cell_type == HR_cell_of_interest)].copy()

    # Sort by to_cell_id and synapses in descending order
    cell_of_interest_inputs_sorted = cell_of_interest_inputs.sort_values(by=['to_cell_id', 'synapses'], ascending=[True, False])

    # Group by to_cell_id and take the first row for each group (highest synapse value)
    HR_unique_highest_inputs = cell_of_interest_inputs_sorted.drop_duplicates(subset='to_cell_id', keep='first')

    #Dropping 'not assigned' rows
    HR_unique_highest_inputs_filtered = HR_unique_highest_inputs[HR_unique_highest_inputs.to_column_id != 'not assigned'].copy()

    # Find the duplicates with the same to_cell_id and synapse value
    HR_duplicates_in_highest_inputs = cell_of_interest_inputs_sorted[cell_of_interest_inputs_sorted.duplicated(subset=['to_cell_id', 'synapses'], keep=False)]

    # Changing column names for clarity
    HR_unique_highest_inputs_filtered.rename(columns={'from_cell_id': 'HR_cell_id', 'from_cell_type': 'HR_cell_type', 'from_column_id': 'HR_column_id', 
                                                    'to_cell_id': 'home_cell_id', 'to_cell_type': 'home_cell_type','to_column_id': 'home_column_id'}, inplace = True)
    # Comvertions to strings
    HR_unique_highest_inputs_filtered['HR_cell_id'] = HR_unique_highest_inputs_filtered['HR_cell_id'].apply(str)


    ## Combining the two correlators

    # Merge data frames of both correlators (with inner join to keep only the rows with keys that are present in both DataFrames)
    HR_BL_unique_highest_inputs_filtered = BL_unique_highest_inputs_filtered.merge(
        HR_unique_highest_inputs_filtered[['HR_cell_id', 'HR_cell_type','HR_column_id','home_cell_id']],
        how='inner',
        left_on='home_cell_id',
        right_on='home_cell_id'
    )

    #Dropping 'not assigned' rows
    HR_BL_unique_highest_inputs_filtered = HR_BL_unique_highest_inputs_filtered[HR_BL_unique_highest_inputs_filtered.BL_column_id != 'not assigned'].copy()
    HR_BL_unique_highest_inputs_filtered = HR_BL_unique_highest_inputs_filtered[HR_BL_unique_highest_inputs_filtered.HR_column_id != 'not assigned'].copy()


    ## Plotting on the 2D lattice: check access to each point and axis
    fig, ax, hexagons = plot_hex_grid(new_centered_p_values, original_q, hex_size=0.5, spacing=1.0, fig_size=(20, 20), labels=labels, label_type='manual_labels', text_size=6)

    # Example to color hexagons based on a condition
    # for p_x, p_y in p:
    #     color_in_p = p_x
    #     color_in_q = p_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_centered_p_values, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightgreen')

    # for q_x, q_y in q:
    #     color_in_p = q_x
    #     color_in_q = q_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_centered_p_values, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightblue')
                
    new_h_x_ls = []
    for h_x, h_y in h:
        color_in_p = h_x
        color_in_p = add_space_in_between(18, 18, _space, [color_in_p])
        color_in_p = color_in_p[0]
        color_in_q = h_y
        for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_centered_p_values, original_q)):
            new_p_pos = calculate_new_p_values([color_in_p], [y_pos], start_key=-16, end_key=17, relative_change=_relative_change)  - center_shift# dealing with shifts
            if x_pos == new_p_pos[0] and y_pos == color_in_q:
                hexagon.set_facecolor('y')
                new_h_x_ls.append(new_p_pos[0])
    
    new_180_0_deg_axis = list(zip(new_h_x_ls,h_y_ls)) # This is my reference line to calculate vectors angles.


    # for v_x, v_y in v:
    #     color_in_p = v_x
    #     color_in_q = v_y
    #     for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_centered_p_values, original_q)):
    #         if x_pos == color_in_p and y_pos == color_in_q:
    #             hexagon.set_facecolor('lightgrey')
                

    '''
    Ideally the nomenclature must relate: "start_ids"  with "from_column_id" and "end_ids" with "to_column_id"
    as written below. However, due to a logic mistake, in a previous step, I run what is written below this comment.
    I will need to fix that later. The current fast fix is correct in the analysis.

    start_ids = [int(x) for x in unique_highest_inputs_filtered.from_column_id.tolist()]
    end_ids = [int(x) for x in unique_highest_inputs_filtered.to_column_id.tolist()]

    '''

    #Preparing the vectors to draw
    if correlator == 'BL':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()] 
        from_cell_of_interest = BL_cell_of_interest
    elif correlator == 'HR':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()]
        from_cell_of_interest = HR_cell_of_interest
    elif correlator == 'HR-BL':
        start_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
        end_ids = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()]  
        from_cell_of_interest = [HR_cell_of_interest,BL_cell_of_interest]

    start_coords_ls = []
    end_coords_ls = []
    for start_id, end_id in zip(start_ids, end_ids):
        start_id_p = df_grid[df_grid.column_id == str(start_id)].new_centered_p.values[0]
        start_id_q = df_grid[df_grid.column_id == str(start_id)].q.values[0]
        start_coords_ls.append((start_id_p,start_id_q))

        
        end_id_p = df_grid[df_grid.column_id == str(end_id)].new_centered_p.values[0]
        end_id_q = df_grid[df_grid.column_id == str(end_id)].q.values[0]
        end_coords_ls.append((end_id_p,end_id_q))

    #Drawing vectors
    for start_coord, end_coord in zip(start_coords_ls,end_coords_ls):
        # Find the coordinates of the start and end hexagons
        x_start = start_coord[0]
        y_start = start_coord[1]
        
        x_end = end_coord[0]
        y_end = end_coord[1]
        
        draw_vector(ax, x_start, y_start, x_end, y_end, color=color_dict[to_cell_of_interest])


    plt.title(f'{correlator} correlator, {to_cell_of_interest}')
    #plt.show()

    # Saving plot
    if save_figures:
        figure_title = f'\Honeycomb_grid_plot_vectors_{from_cell_of_interest}-home-{to_cell_of_interest}_correlator-{correlator}.pdf'
        fig.savefig(save_path+figure_title)
        print('FIGURE: Visualization vectors on the 2D grid plotted')
    plt.close(fig)




    ## Quantifying angles realtuve to the qeuatorial axis (h)

    # Given points defining the new line
    points = new_180_0_deg_axis

    # Calculate the slope (m) and angle of the new line
    x1, y1 = points[0]
    x2, y2 = points[-1]
    slope = (y2 - y1) / (x2 - x1)
    reference_angle_rad = np.arctan(slope)

    # Prepare the list for angles
    angles = []

    # Calculate the angle for each vector
    for start_id, end_id, start_coord, end_coord in zip(start_ids, end_ids, start_coords_ls, end_coords_ls):
        # Coordinates
        x_start, y_start = start_coord
        x_end, y_end = end_coord
        
        # Calculate the change in coordinates
        dx = x_end - x_start
        dy = y_end - y_start
        
        # Calculate the angle in radians relative to the x-axis
        angle_rad = np.arctan2(dy, dx)
        
        # Adjust the angle relative to the new reference line
        adjusted_angle_rad = angle_rad - reference_angle_rad
        
        # Convert to degrees
        adjusted_angle_deg = np.degrees(adjusted_angle_rad)
        
        # Normalize the angle to be in the range [0, 360) degrees
        adjusted_angle_deg = (adjusted_angle_deg + 360) % 360
        
        # Append the result to the angles list
        angles.append((start_id, end_id, adjusted_angle_deg))

    # Create a DataFrame with the results
    angles_df = pd.DataFrame(angles, columns=['start_id', 'end_id', 'angle'])
    angles_df_dict[to_cell_of_interest] = angles_df


#%% Looking at distributions in the honeycomb-like grid. Plotting in a loop
# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for cell in to_cell_of_interest_ls:
    # Data
    to_cell_of_interest = cell
    angles = angles_df_dict[to_cell_of_interest]['angle'] 
    angles_rad = np.radians(angles)  # Convert angles to radians  

    # Normalize angles to the range -π to π
    angles_rad_norm = np.mod(angles_rad + np.pi, 2 * np.pi) - np.pi  

    # KDE Plot
    sns.kdeplot(angles_rad_norm, ax=axs[0], color=color_dict[to_cell_of_interest], label=to_cell_of_interest)

    # Polar Plot
    axs[1] = plt.subplot(122, projection='polar')
    axs[1].hist(angles_rad, bins=30, density=False, edgecolor='black', color=color_dict[to_cell_of_interest], alpha=0.7, histtype='stepfilled', label=to_cell_of_interest)

# Set titles and labels outside the loop
axs[0].set_title(f'KDE of Angles \n{correlator} correlator')
axs[0].set_xlabel('Angle (radians)')
axs[0].set_ylabel('Density')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
axs[0].set_xticks(ticks)
axs[0].set_xticklabels(tick_labels)
axs[0].legend()

axs[1].set_title('Polar Plot of Angles')
axs[1].legend()

plt.tight_layout()
plt.suptitle(f'{correlator} correlator, {", ".join(to_cell_of_interest_ls)}', y=1.02)

# Saving plot
if save_figures:
    figure_title = f'\Vectors_KDE_histogram_{from_cell_of_interest}-home_T4_correlator-{correlator}_from_honeycomb_grid.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of vectors angles plotted')
plt.close(fig)


#%% Looking at distributions in the honeycomb-like grid. Plotting in a loop
# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for cell in to_cell_of_interest_ls:
    # Data
    to_cell_of_interest = cell
    angles = angles_df_dict[to_cell_of_interest]['angle'] 
    angles_rad = np.radians(angles)  # Convert angles to radians  

    # Normalize angles to the range -π to π
    angles_rad_norm = np.mod(angles_rad + np.pi, 2 * np.pi) - np.pi  

    # Histogram Plot
    axs[0].hist(angles_rad_norm, bins=30, edgecolor='black', color=color_dict[to_cell_of_interest], alpha = 0.5, label=to_cell_of_interest)
    axs[0].set_title(f'Histogram of Angles \n{correlator} correlator')
    axs[0].set_xlabel('Angle (degrees)')
    axs[0].set_ylabel('Frequency')

    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(tick_labels)
    axs[0].legend()

    # Polar Plot
    angles_rad = np.radians(angles)  # Convert angles to radians
    axs[1] = plt.subplot(122, projection='polar')
    axs[1].hist(angles_rad, bins=30, density=False, edgecolor='black', color=color_dict[to_cell_of_interest], alpha=0.7, histtype='stepfilled',label=to_cell_of_interest)
    axs[1].legend()

        # Calculate the KDE
    kde = gaussian_kde(angles_rad, bw_method='scott')  # You can adjust the bandwidth method as needed
    theta = np.linspace(0, 2 * np.pi, 1000)  # 1000 points from 0 to 2*pi
    kde_values = kde(theta)

    # Normalize and scale KDE values for better visual representation
    kde_values_scaled = kde_values / np.max(kde_values) * np.max(angles_rad) * 50

    # Plot the KDE on the polar plot
    axs[1].plot(theta, kde_values_scaled, color=color_dict[to_cell_of_interest], lw=2)
    axs[1].set_title('Polar Plot of Angles')

    # Show plots
    plt.tight_layout()
    plt.title(f'{correlator} correlator, {to_cell_of_interest_ls}')
#plt.show()

# Saving plot
if save_figures:
    figure_title = f'\Vectors_histogram_{from_cell_of_interest}-home_T4_correlator-{correlator}_from_honeycomb_grid.pdf'
    fig.savefig(save_path+figure_title)
    print('FIGURE: Visualization of vectors angles plotted')
plt.close(fig)