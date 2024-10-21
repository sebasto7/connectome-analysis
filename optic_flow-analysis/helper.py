# -*- coding: utf-8 -*-
"""
Created on Mon September 26 10:28:00 2024

@author: smolina

helper file with analisis and plotting functions for optic flow analysis of DS cells in FAFB and Janelia male optic lobe data sets
"""

#%% Importing some packages
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
#%% Analsis functions

def combine_xyz(df):
    """
    Combines separated x, y and z column into one, changes units and adds new column names for
    generating a neuroglancer link with function nglui.statebuilder.helpers.make_synapse_neuroglancer_link

    Args:
        pandas data frame containing x,y and z as columns of the same length

    Returns:
        same pandas data frame containing a new column with [x/4,y/4,z/40] lists
    """
    # Generating the single column

    post_pt_position = []
    for x,y,z in zip(df['post_x'].tolist(),df['post_y'].tolist(),df['post_z'].tolist()):
        temp_ls = [x/4,y/4,z/40]
        post_pt_position.append(temp_ls)

    pre_pt_position = []
    for x,y,z in zip(df['pre_x'].tolist(),df['pre_y'].tolist(),df['pre_z'].tolist()):
        temp_ls = [x/4,y/4,z/40]
        pre_pt_position.append(temp_ls)

    #Adding new columns and names
    df['post_pt_position'] = post_pt_position
    df['pre_pt_position'] = pre_pt_position
    #Changing column names
    df.rename(columns={'pre': 'pre_pt_root_id', 'post': 'post_pt_root_id'}, inplace=True)

def calculate_new_p_values(original_x, original_y, start_key=-16, end_key=17, relative_change=0.5, in_space=0):
    """
    Calculates new x values by applying a relative change based on a mapping of y values.

    Parameters:
    -----------
    original_x : list or array-like
        The original x values to be adjusted.
    original_y : list or array-like
        The corresponding y values used to determine the adjustment.
    start_key : int, optional
        The starting key for the range of y values to be mapped. Default is -16.
    end_key : int, optional
        The ending key for the range of y values to be mapped. Default is 17.
    relative_change : float, optional
        The factor by which x values will be adjusted. Default is 0.5.
    in_space : int, optional
        An optional parameter for additional spacing or modification. Default is 0.

    Returns:
    --------
    new_x_values : list
        A list of new x values after applying the relative change based on y values.
    """
    shift_dict = {}

    for i in range(start_key, end_key + 1):
        value = (i - start_key) * relative_change
        shift_dict[i] = value

    new_x_values = []

    for x, y in zip(original_x, original_y):
        if y in shift_dict:
            relative_change = shift_dict[y]
            new_x = x - relative_change
            new_x_values.append(new_x)
        else:
            new_x_values.append(x)

    return new_x_values


def add_space_in_between(min_max_coordinates_dict,parameters_dict, key_list):
    
    """
    Adds space between negative and positive values and returns a new list based on a provided key list.

    Parameters:
    -----------
    num_negatives : int
        The number of negative values to generate.
    num_positives : int
        The number of positive values to generate.
    space : float
        The spacing factor between the values.
    key_list : list
        A list of keys to generate the corresponding spaced values.

    Returns:
    --------
    new_list : list
        A list of values generated based on the key list with added spacing.
    """

    #Unpacking values from dictionaries
    num_negatives = min_max_coordinates_dict['min_p_value'] 
    num_positives = min_max_coordinates_dict['max_p_value'] 
    space = parameters_dict['space']

    #Main functionality
    num_negatives = num_negatives*-1
    negative_numbers = [-space * i for i in range(num_negatives, 0, -1)]
    positive_numbers = [space * i for i in range(1, num_positives + 1)]
    generated_list = negative_numbers + [0] + positive_numbers
    original_range = list(range(-num_negatives, num_positives + 1))
    
    space_dict = {}
    for i, key in enumerate(original_range):
        if i < len(generated_list):
            space_dict[key] = generated_list[i]
        else:
            space_dict[key] = None

    new_list = [space_dict.get(key, None) for key in key_list]
    
    return new_list

def add_space_in_between_2(min_max_coordinates_dict, parameters_dict, key_list):
    
    """
    Adds space between negative and positive values and returns a new list based on a provided key list.

    Parameters:
    -----------
    min_max_coordinates_dict : dict
        A dictionary containing the 'min_p_value' and 'max_p_value'.
    parameters_dict : dict
        A dictionary containing the 'space' value.
    key_list : list
        A list of keys to generate the corresponding spaced values.

    Returns:
    --------
    new_list : list
        A list of values generated based on the key list with added spacing.
    """

    # Unpacking values from dictionaries
    num_negatives = min_max_coordinates_dict['min_p_value']
    num_positives = min_max_coordinates_dict['max_p_value']
    space = parameters_dict['space']

    # Main functionality
    # If num_negatives or num_positives are float, we'll need to generate the range manually
    if isinstance(num_negatives, float) or isinstance(num_positives, float):
        # Generating float-based ranges using linspace for negative and positive ranges
        negative_numbers = np.linspace(-space * num_negatives, -space, int(num_negatives))
        positive_numbers = np.linspace(space, space * num_positives, int(num_positives))
    else:
        # When both values are integers, we can use the original logic
        num_negatives = num_negatives * -1  # Convert to negative for the range
        negative_numbers = [-space * i for i in range(num_negatives, 0, -1)]
        positive_numbers = [space * i for i in range(1, num_positives + 1)]
    
    # Combine both negative and positive numbers with a 0 in between
    generated_list = list(negative_numbers) + [0] + list(positive_numbers)

    # Create the original range, treating negative to positive values
    original_range = list(range(-int(num_negatives), int(num_positives) + 1))

    # Mapping the original range to the generated list
    space_dict = {}
    for i, key in enumerate(original_range):
        if i < len(generated_list):
            space_dict[key] = generated_list[i]
        else:
            space_dict[key] = None

    # Build the final list based on the provided key list
    new_list = [space_dict.get(key, None) for key in key_list]
    
    return new_list

def getting_correlators_start_end_coordinates(corr_ls, df_grid, HR_BL_unique_highest_inputs_filtered, map_type):
    for corr_i in corr_ls:
        if corr_i == 'BL':
            start_ids = HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()
            end_ids = HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist() 
        elif corr_i == 'HR':
            start_ids = HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()
            end_ids = HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()  
        elif corr_i == 'HR-BL':
            start_ids = HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()
            end_ids = HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()  

        # Convert to integers
        start_ids = [int(x) for x in start_ids]
        end_ids = [int(x) for x in end_ids]

        if map_type == 'hexagonal':
            # Getting vector coordinates    
            start_coords_ls = []
            end_coords_ls = []
            for start_id, end_id in zip(start_ids, end_ids):
                start_id_p = df_grid[df_grid.column_id == str(start_id)].new_centered_p.values[0]
                start_id_q = df_grid[df_grid.column_id == str(start_id)].q.values[0]
                start_coords_ls.append((start_id_p, start_id_q))
    
                end_id_p = df_grid[df_grid.column_id == str(end_id)].new_centered_p.values[0]
                end_id_q = df_grid[df_grid.column_id == str(end_id)].q.values[0]
                end_coords_ls.append((end_id_p, end_id_q))
        elif map_type == 'regular':
            # Getting vector coordinates    
            start_coords_ls = []
            end_coords_ls = []
            for start_id, end_id in zip(start_ids, end_ids):
                start_id_p = df_grid[df_grid.column_id == str(start_id)].p.values[0]
                start_id_q = df_grid[df_grid.column_id == str(start_id)].q.values[0]
                start_coords_ls.append((start_id_p, start_id_q))
    
                end_id_p = df_grid[df_grid.column_id == str(end_id)].p.values[0]
                end_id_q = df_grid[df_grid.column_id == str(end_id)].q.values[0]
                end_coords_ls.append((end_id_p, end_id_q))

        elif map_type == 'regular-rotated':
            # Getting vector coordinates    
            start_coords_ls = []
            end_coords_ls = []
            for start_id, end_id in zip(start_ids, end_ids):
                start_id_p = df_grid[df_grid.column_id == str(start_id)].rotated_p.values[0]
                start_id_q = df_grid[df_grid.column_id == str(start_id)].rotated_q.values[0]
                start_coords_ls.append((start_id_p, start_id_q))
    
                end_id_p = df_grid[df_grid.column_id == str(end_id)].rotated_p.values[0]
                end_id_q = df_grid[df_grid.column_id == str(end_id)].rotated_q.values[0]
                end_coords_ls.append((end_id_p, end_id_q))
            

        # Saving start and end IDs for BL, HR, and HR-BL
        if corr_i == 'BL':
            HR_BL_unique_highest_inputs_filtered['BL_start_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()]
            HR_BL_unique_highest_inputs_filtered['BL_end_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()] 
        
        elif corr_i == 'HR':
            HR_BL_unique_highest_inputs_filtered['HR_start_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
            HR_BL_unique_highest_inputs_filtered['HR_end_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.home_column_id.tolist()] 

        elif corr_i == 'HR-BL': 
            HR_BL_unique_highest_inputs_filtered['HR-BL_start_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.HR_column_id.tolist()]
            HR_BL_unique_highest_inputs_filtered['HR-BL_end_ids'] = [int(x) for x in HR_BL_unique_highest_inputs_filtered.BL_column_id.tolist()]  

        # Saving coordinates in the main data frame
        HR_BL_unique_highest_inputs_filtered[f'{corr_i}_start_coords'] = start_coords_ls
        HR_BL_unique_highest_inputs_filtered[f'{corr_i}_end_coords'] = end_coords_ls

def getting_correlators_angles(points, HR_BL_unique_highest_inputs_filtered, corr_ls):
    # Given points defining the new line
    x1, y1 = points[0]
    x2, y2 = points[-1]
    
    # Calculate the slope (m) and angle of the new line
    slope = (y2 - y1) / (x2 - x1)
    reference_angle_rad = np.arctan(slope)

    for corr_i in corr_ls:
        # Prepare the list for angles and other variables
        angles = []
        start_ids = HR_BL_unique_highest_inputs_filtered[f'{corr_i}_start_ids'].tolist()
        end_ids = HR_BL_unique_highest_inputs_filtered[f'{corr_i}_end_ids'].tolist()
        start_coords_ls = HR_BL_unique_highest_inputs_filtered[f'{corr_i}_start_coords'].tolist()
        end_coords_ls = HR_BL_unique_highest_inputs_filtered[f'{corr_i}_end_coords'].tolist()
        
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
            adjusted_angle_deg = round(adjusted_angle_deg)  # rounding angle's values
            
            # Append the result to the angles list
            angles.append((start_id, end_id, adjusted_angle_deg))
        
        # Create a DataFrame with the results
        angles_df = pd.DataFrame(angles, columns=['start_id', 'end_id', 'angle']) 
        
        # Updating the main data frame with angles
        HR_BL_unique_highest_inputs_filtered[f'{corr_i}_angle'] = angles_df['angle'].tolist()
    
    return HR_BL_unique_highest_inputs_filtered  # Returning the modified DataFrame


def get_input_fractions_and_norm(ol_connections, HR_BL_unique_highest_inputs_filtered):
    ## Getting all synaptic data 
    total_synapse_num = ol_connections.groupby('to_cell_id')['synapses'].agg('sum')

    ## Getting input weights
    # Map the total synapse numbers to the corresponding 'home_cell_id' in the filtered DataFrame
    HR_BL_unique_highest_inputs_filtered['total_input_synapses'] = HR_BL_unique_highest_inputs_filtered['home_cell_id'].map(total_synapse_num)

    # Calculate the actual input fractions
    HR_BL_unique_highest_inputs_filtered['BL_fraction'] = HR_BL_unique_highest_inputs_filtered['BL_synapses'] / HR_BL_unique_highest_inputs_filtered['total_input_synapses']
    HR_BL_unique_highest_inputs_filtered['HR_fraction'] = HR_BL_unique_highest_inputs_filtered['HR_synapses'] / HR_BL_unique_highest_inputs_filtered['total_input_synapses']

    # Normalizing absolute synapse number
    _max_norm_BL_synapses_ls = HR_BL_unique_highest_inputs_filtered['BL_synapses'] / HR_BL_unique_highest_inputs_filtered['BL_synapses'].max()
    HR_BL_unique_highest_inputs_filtered['max_norm_BL_synapses'] = _max_norm_BL_synapses_ls

    _max_norm_HR_synapses_ls = HR_BL_unique_highest_inputs_filtered['HR_synapses'] / HR_BL_unique_highest_inputs_filtered['HR_synapses'].max()
    HR_BL_unique_highest_inputs_filtered['max_norm_HR_synapses'] = _max_norm_HR_synapses_ls

    HR_BL_unique_highest_inputs_filtered['HR-BL_synapses'] = HR_BL_unique_highest_inputs_filtered['BL_synapses'] + HR_BL_unique_highest_inputs_filtered['HR_synapses']
    _max_norm_HR_BL_synapses_ls = HR_BL_unique_highest_inputs_filtered['HR-BL_synapses'] / HR_BL_unique_highest_inputs_filtered['HR-BL_synapses'].max()
    HR_BL_unique_highest_inputs_filtered['max_norm_HR-BL_synapses'] = _max_norm_HR_BL_synapses_ls

    # Normalizing input weights
    _max_norm_BL_input_fraction_ls = HR_BL_unique_highest_inputs_filtered['BL_fraction'] / HR_BL_unique_highest_inputs_filtered['BL_fraction'].max()
    HR_BL_unique_highest_inputs_filtered['max_norm_BL_input_fraction'] = _max_norm_BL_input_fraction_ls

    _max_norm_HR_input_fraction_ls = HR_BL_unique_highest_inputs_filtered['HR_fraction'] / HR_BL_unique_highest_inputs_filtered['HR_fraction'].max()
    HR_BL_unique_highest_inputs_filtered['max_norm_HR_input_fraction'] = _max_norm_HR_input_fraction_ls

    # Calculate input weights considering only HR (Mi9) and BL (Mi4) inputs for the total number of synapses
    HR_BL_sum = HR_BL_unique_highest_inputs_filtered['BL_synapses'] + HR_BL_unique_highest_inputs_filtered['HR_synapses']
    HR_BL_unique_highest_inputs_filtered['BL_input_fraction_HR-BL'] = HR_BL_unique_highest_inputs_filtered['BL_synapses'] / HR_BL_sum 
    HR_BL_unique_highest_inputs_filtered['HR_input_fraction_HR-BL'] = HR_BL_unique_highest_inputs_filtered['HR_synapses'] / HR_BL_sum 



def get_weighted_correlator(HR_BL_unique_highest_inputs_filtered):
    # Based on each correlator input fraction concerning each other, calculate the weighted vector sum
    df = HR_BL_unique_highest_inputs_filtered
    HR_weight_variable = 'HR_input_fraction_HR-BL'
    BL_weight_variable = 'BL_input_fraction_HR-BL'

    # Convert angles from degrees to radians
    df['BL_angle_rad'] = np.deg2rad(df['BL_angle'])
    df['HR_angle_rad'] = np.deg2rad(df['HR_angle'])

    # Calculate x and y components of BL and HR vectors
    df['BL_x'] = df[BL_weight_variable] * np.cos(df['BL_angle_rad'])
    df['BL_y'] = df[BL_weight_variable] * np.sin(df['BL_angle_rad'])
    df['HR_x'] = df[HR_weight_variable] * np.cos(df['HR_angle_rad'])
    df['HR_y'] = df[HR_weight_variable] * np.sin(df['HR_angle_rad'])

    # Compute resultant x and y components by summing the BL and HR components
    HR_BL_unique_highest_inputs_filtered['resultant_x'] = df['BL_x'] + df['HR_x']
    HR_BL_unique_highest_inputs_filtered['resultant_y'] = df['BL_y'] + df['HR_y']

    # Calculate the average x and y components
    avg_resultant_x = HR_BL_unique_highest_inputs_filtered['resultant_x'].mean()
    avg_resultant_y = HR_BL_unique_highest_inputs_filtered['resultant_y'].mean()

    # Convert the average x and y components back to polar coordinates
    avg_resultant_radius = np.sqrt(avg_resultant_x**2 + avg_resultant_y**2)
    avg_resultant_angle = np.arctan2(avg_resultant_y, avg_resultant_x)

    # Convert resultant vector to polar form (magnitude and angle)
    HR_BL_unique_highest_inputs_filtered['HR-BL_weighted_vector_magnitude'] = np.sqrt(HR_BL_unique_highest_inputs_filtered['resultant_x']**2 + HR_BL_unique_highest_inputs_filtered['resultant_y']**2)
    HR_BL_unique_highest_inputs_filtered['HR-BL_weighted_vector_angle'] = np.rad2deg(np.arctan2(HR_BL_unique_highest_inputs_filtered['resultant_y'], HR_BL_unique_highest_inputs_filtered['resultant_x']))

    # Apply the function to each row to get end coordinates for weighted HR-BL vectors
    # Function to compute the resultant end coordinates
    def compute_end_coords(row):
        start_x, start_y = row['HR-BL_start_coords']  # Unpacking the start coordinates
        resultant_x = row['resultant_x']
        resultant_y = row['resultant_y']
        
        # Compute the end coordinates
        end_x = start_x + (resultant_x*2.5)
        end_y = start_y + (resultant_y*2.5)
        
        return (end_x, end_y)
        
    HR_BL_unique_highest_inputs_filtered['HR-BL_resultant_end_coords'] = HR_BL_unique_highest_inputs_filtered.apply(compute_end_coords, axis=1)

    return avg_resultant_radius, avg_resultant_angle

# Function to apply rotation to a point (x, y)
def rotate_point(x, y, cos_theta, sin_theta):
    x_new = x * cos_theta + y * sin_theta
    y_new = -x * sin_theta + y * cos_theta
    return (x_new, y_new)



def get_hexagons_reference_equators(reference_axes_dict, map_type, x_start = -30, x_steps = 50):
    """
    
    Add Docstring once the function is complete
    
    """
    # Needed variables
    h_x_ls = reference_axes_dict['h_x_ls']
    h_x_ls = [x+x_start for x in h_x_ls]
    h_y_ls = reference_axes_dict['h_y_ls'] 

    equators_list = []
    for i in range(x_steps):
        h_x_ls = [x+1 for x in h_x_ls]
        # Coloruing the hexagons
        h = list(zip(h_x_ls,h_y_ls))
        equators_list.append(h)
    return equators_list

def match_pq_to_equators(df, equators_list):
    """
    Match the (p, q) values from the DataFrame to the list of equator tuples
    in equators_list and label the matched row with the index of the matched sublist.
    """
    # Initialize a new column to store the matching equator index
    df['equator_match'] = None
    
    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        p_val = row['p']
        q_val = row['q']
        pq_tuple = (p_val, q_val)
        
        # Search for the tuple in the equators_list
        for eq_index, equator in enumerate(equators_list):
            if pq_tuple in equator:
                df.at[index, 'equator_match'] = equator  # Store the equator of the matching equator
                break  # Exit the loop once a match is found
    
    return df

def add_equator_match_ids(df):
    """
    Adds a new column 'equator_match_ids' to the DataFrame containing
    column_ids that match the (p, q) coordinates in the equator_match column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    
    Returns:
    pd.DataFrame: The modified DataFrame with the new column.
    """
    # Initialize a list to store the equator match IDs
    equator_match_ids = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the (p, q) combinations from equator_match
        equator_matches = row['equator_match']
        matching_ids = set()  # Use a set to avoid duplicates
        
        # Check each (p, q) combination
        for p_val, q_val in equator_matches:
            # Find matching rows in df
            matches = df[(df['p'] == p_val) & (df['q'] == q_val)]
            
            # Add the column_id of each matching row to the set
            for _, match_row in matches.iterrows():
                matching_ids.add(match_row['column_id'])

        # Convert the set to a list and store it in the equator_match_ids list
        equator_match_ids.append(list(matching_ids))

    # Assign the list of equator match IDs to a new column in the DataFrame
    df['equator_match_ids'] = equator_match_ids
    return df


    
def find_nearest_neighbors(df, n_neighbors=24):
    from sklearn.neighbors import NearestNeighbors
    """
    Find the nearest neighbors for each row based on centroid_x, centroid_y, and centroid_z.
    Add a new column 'nearest_neighbours' with the list of the 24 nearest neighbors' column_id.
    """
    # Extract the coordinates (centroid_x, centroid_y, centroid_z)
    coordinates = df[['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Initialize the NearestNeighbors model (including self in the neighbors)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(coordinates)
    
    # Find the indices of the nearest neighbors (including self, so we use n_neighbors+1)
    distances, indices = nbrs.kneighbors(coordinates)
    
    # Add the nearest neighbor column (excluding self which is the first neighbor)
    df['nearest_neighbours'] = indices[:, 1:].tolist()  # Skip the first neighbor (self)
    
    # Map indices to the corresponding 'column_id' values for neighbors
    df['nearest_neighbours'] = df['nearest_neighbours'].apply(lambda idx_list: df.iloc[idx_list]['column_id'].tolist())
    
    return df



def project_to_2d_plane(df):
    from sklearn.decomposition import PCA
    """
    Project the centroid_x, centroid_y, centroid_z of each row and its nearest neighbors 
    onto a 2D plane defined by the first two principal components (PC1 and PC2) using PCA.
    
    Parameters:
    - df: DataFrame containing the data with columns ['column_id', 'centroid_x', 'centroid_y', 'centroid_z', 'nearest_neighbours']
    
    Returns:
    - DataFrame with new columns:
        - 'local_plane_coords': containing the projected coordinates of the row and its neighbors as lists.
        - 'home_dot_coords': containing the 2D projected coordinates of the current row.
    """
    # Initialize the new columns
    df['local_plane_coords'] = None
    df['home_dot_coords'] = None
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the current row's column_id and its nearest neighbors' ids
        current_column_id = row['column_id']
        nearest_neighbors_ids = row['nearest_neighbours']
        
        # Get the relevant rows for PCA: current row and its nearest neighbors
        current_row = df[df['column_id'] == current_column_id]
        neighbor_rows = df.set_index('column_id').loc[nearest_neighbors_ids].reset_index()
        
        # Reorder rows to have the current row first
        relevant_rows_ordered = pd.concat([current_row, neighbor_rows])
        
        # Check if we have enough rows for PCA
        if len(relevant_rows_ordered) < 2:
            print(f"Not enough rows to perform PCA for Column ID {current_column_id}.")
            continue
        
        # Prepare data for PCA
        coordinates = relevant_rows_ordered[['centroid_x', 'centroid_y', 'centroid_z']].values
        
        # Apply PCA
        pca = PCA(n_components=2)
        projected_coords = pca.fit_transform(coordinates)
        
        # Save the projected coordinates of all points in 'local_plane_coords'
        neightbors_dots_projected = projected_coords[1:]
        df.at[index, 'local_plane_coords'] = neightbors_dots_projected.tolist()
        
        # The first row corresponds to the current row (home dot)
        home_dot_projected = projected_coords[0]  # Now this is guaranteed to be the current row
        
        # Save the home dot projected coordinates in 'home_dot_coords'
        df.at[index, 'home_dot_coords'] = home_dot_projected.tolist()
    
    return df

# def project_vector_to_2d_plane(df):
#     def project_vector(row):
#         # Extract the relevant coordinates from the row
#         local_plane_coords = np.array(row['local_plane_coords'])
#         center_mi1 = np.array([row['center_mi1_x'], row['center_mi1_y'], row['center_mi1_z']])
#         center_mi4 = np.array([row['center_mi4_x'], row['center_mi4_y'], row['center_mi4_z']])
        
#         # Create the start and end points for the vector in 2D
#         start_2d = center_mi1[:2]  # Take x and y from center_mi1
#         end_2d = center_mi4[:2]    # Take x and y from center_mi4
        
#         # Store the vector's start and end position as a list in the new column
#         return [start_2d.tolist(), end_2d.tolist()]
    
#     # Apply the projection function to each row and create a new column
#     df['vector_dots_coords'] = df.apply(project_vector, axis=1)
    
#     return df


# def project_vector_to_defined_plane(df):
#     """
#     Projects 3D vector coordinates onto a defined 2D plane using the normal vector of the plane.

#     This function takes a DataFrame containing 3D coordinates and a set of 2D coordinates that define 
#     a plane. It computes the projection of the start and end points of each vector (defined by 
#     `center_mi1` and `center_mi4`) onto the specified plane.

#     Parameters:
#     ----------
#     df : pandas.DataFrame
#         A DataFrame containing the following columns:
#         - 'local_plane_coords': A list of points defining the plane (should contain at least three points).
#         - 'center_mi1_x', 'center_mi1_y', 'center_mi1_z': The x, y, z coordinates for the start point of the vector.
#         - 'center_mi4_x', 'center_mi4_y', 'center_mi4_z': The x, y, z coordinates for the end point of the vector.

#     Returns:
#     -------
#     pandas.DataFrame
#         The original DataFrame with an additional column named 'vector_dots_coords', which contains 
#         the projected x and y coordinates of the start and end points of the vectors on the defined plane.
#         Each entry in this column is a dictionary with keys 'start' and 'end' containing the 
#         respective projected 2D coordinates.
#     """
    
#     def project_to_plane(point, plane_point, normal):
#         """Project a 3D point onto the defined plane."""
#         # Vector from the plane point to the point to project
#         point_to_plane = point - plane_point
        
#         # Calculate the dot product to find the projection length
#         distance_to_plane = np.dot(point_to_plane, normal) / np.linalg.norm(normal)**2
        
#         # Project the point onto the plane
#         projected_point = point - distance_to_plane * normal
#         return projected_point

#     def compute_plane_normal(points):
#         """Compute the best-fit plane normal from a set of points using SVD."""
#         # Center the points
#         points = np.array(points)
#         centroid = np.mean(points, axis=0)
#         centered_points = points - centroid
        
#         # Perform Singular Value Decomposition
#         _, _, vh = np.linalg.svd(centered_points)
        
#         # The normal vector is the last column of V (vh)
#         normal = vh[-1]
#         return normal, centroid

#     def project_vector(row):
#         # Extract coordinates
#         local_plane_coords = np.array(row['local_plane_coords'])
#         center_mi1 = np.array([row['center_mi1_x'], row['center_mi1_y'], row['center_mi1_z']])
#         center_mi4 = np.array([row['center_mi4_x'], row['center_mi4_y'], row['center_mi4_z']])
        
#         # Compute the normal vector of the plane defined by all local plane coordinates
#         normal, plane_point = compute_plane_normal(local_plane_coords)
        
#         # Ensure plane_point is a 3D point
#         plane_point = np.array([plane_point[0], plane_point[1], 0])  # Assuming the plane is at z=0
        
#         # Project the start and end points onto the defined plane
#         projected_start = project_to_plane(center_mi1, plane_point, normal)
#         projected_end = project_to_plane(center_mi4, plane_point, normal)
        
#         # Return only the x and y coordinates
#         return {
#             'start': projected_start[:2].tolist(),  # x and y of projected start
#             'end': projected_end[:2].tolist()       # x and y of projected end
#         }
    
#     # Apply the projection function to each row and create a new column
#     df['vector_dots_coords'] = df.apply(project_vector, axis=1)
    
#     return df



def fit_reference_line_to_plane(df_grid_extended, _column_id):
    """
    Function to extract data and compute the best fit line for a given column_id.
    
    Parameters:
    df_grid_extended (pd.DataFrame): DataFrame containing the grid data.
    _column_id (str): The column ID for which the data is extracted.

    Returns:
    x_plane, y_plane (arrays): Coordinates of the projected points.
    x_home, y_home (floats): Coordinates of the home dot.
    x_fit, y_fit (arrays): Coordinates for the best fit line.
    """
    # Extract nearest neighbors and equator match IDs for a given column_id
    chosen_row = df_grid_extended[df_grid_extended['column_id'] == _column_id]
    local_plane_coords_row = chosen_row.local_plane_coords

    if chosen_row.empty:
        raise ValueError(f"Column ID '{_column_id}' not found.")
    
    nearest_neighbors = chosen_row['nearest_neighbours'].values[0]
    equator_match_ids = chosen_row['equator_match_ids'].values[0]

    # Convert equator_match_ids to a set for faster lookup
    equator_match_ids_set = set(equator_match_ids)

    # Find positions of nearest neighbors that are in equator_match_ids
    positions = [i for i, neighbor in enumerate(nearest_neighbors) if neighbor in equator_match_ids_set]

    reference_equator_coord_2D = []
    for pos in positions:
        temp_coord = local_plane_coords_row.values.tolist()[0][pos]
        reference_equator_coord_2D.append(temp_coord)

    # Add the position of the chosen column id
    home_dot_coord = chosen_row.home_dot_coords.values.tolist()[0]
    home_dot_coord_array = np.array(home_dot_coord)
    reference_equator_coord_2D = np.vstack([reference_equator_coord_2D, home_dot_coord_array])

    # Sample data from df_grid_extended for 'given' column_id 
    local_plane_coords = np.array(df_grid_extended[df_grid_extended.column_id == _column_id].local_plane_coords.tolist()[0])
    local_home_coords = np.array(df_grid_extended[df_grid_extended.column_id == _column_id].home_dot_coords.tolist()[0])
    x_plane = local_plane_coords[:, 0]
    y_plane = local_plane_coords[:, 1]
    x_home = local_home_coords[0]
    y_home = local_home_coords[1]

    # Reference equator coordinates (example values or calculated previously)
    reference_equator_coord_2D = np.array(reference_equator_coord_2D)

    # Extract x and y coordinates for analysis
    x_equator = reference_equator_coord_2D[:, 0]
    y_equator = reference_equator_coord_2D[:, 1]

    # Calculate the absolute range for both axes
    x_range = np.max(np.abs(x_equator)) - np.min(np.abs(x_equator))
    y_range = np.max(np.abs(y_equator)) - np.min(np.abs(y_equator))

    # Determine which axis has the largest spread and fit a line accordingly
    if y_range > x_range:
        # Fit a line based on the y-axis (larger range)
        slope, intercept = np.polyfit(y_equator, x_equator, 1)
        y_fit = np.linspace(np.min(y_equator), np.max(y_equator), 100)
        x_fit = slope * y_fit + intercept
    else:
        # Fit a line based on the x-axis (larger range)
        slope, intercept = np.polyfit(x_equator, y_equator, 1)
        x_fit = np.linspace(np.min(x_equator), np.max(x_equator), 100)
        y_fit = slope * x_fit + intercept

    return x_plane, y_plane, x_home, y_home, x_fit, y_fit

def calculate_bl_angles(df, columns_id_data):
    angles = []
    # Iterate over each row in the dataframe
    for idx, row in df.iterrows():
        # IDs for Mi4 and Mi1
        temp_Mi4_cell_ID = row['Mi4_ID']
        temp_Mi1_cell_ID = row['Mi1_ID']

        # Get column IDs using the 'columns_id_data' DataFrame
        temp_Mi4_column_ID = columns_id_data[columns_id_data['root_id'] == temp_Mi4_cell_ID]['column_id'].tolist()[0]
        temp_Mi1_column_ID = columns_id_data[columns_id_data['root_id'] == temp_Mi1_cell_ID]['column_id'].tolist()[0]

        # Vector Start and End coordinates (Home Dot and Mi4 Nearest Neighbor)
        temp_neighbors_ls = row['nearest_neighbours']
        try:
            temp_index = temp_neighbors_ls.index(temp_Mi4_column_ID)
            temp_end_coords = row['local_plane_coords'][temp_index]
            temp_start_coords = row['home_dot_coords']
        except:
            # Vectors with no length should not be considered.
            temp_end_coords = row['home_dot_coords']
            temp_start_coords = row['home_dot_coords']
            angles.append(None)
            continue  # Skip to the next row

        # Fit the reference line and get coordinates for the plane (best fit line)
        x_plane, y_plane, x_home, y_home, x_fit, y_fit = fit_reference_line_to_plane(df, row['column_id'])

        # Given points defining the new line
        x1 = x_fit[0]
        y1 = y_fit[0]
        x2 = x_fit[-1]
        y2 = y_fit[-1]
        
        # Calculate the slope (m) and angle of the new line
        slope = (y2 - y1) / (x2 - x1)
        reference_angle_rad = np.arctan(slope)
    
            
        # Calculate the vector angle
        x_start = temp_start_coords[0]
        y_start = temp_start_coords[1]
        x_end = temp_end_coords[0]
        y_end = temp_end_coords[1]
        
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
        adjusted_angle_deg = round(adjusted_angle_deg)  # rounding angle's values
        
        # Append the result to the angles list
        angles.append(adjusted_angle_deg)
    
    
    # Updating the main data frame with angles
    df[f'BL_angle'] = angles

    return df # Returning the modified DataFrame


#%% Plotting functions

def plot_tetra_grid(x, y, tet_size=1.0, spacing=1.5, fig_size=(10, 10), labels=None, label_type='column_id', text_size=10, ax=None):
    """
    Plots a tetragonal grid using the provided x and y coordinates.

    Parameters:
    -----------
    x : list or array-like
        List of x coordinates for the centers of the tetragons.
    y : list or array-like
        List of y coordinates for the centers of the tetragons.
    tet_size : float, optional
        The size of each tetragon, defined as the distance from the center to any vertex. Default is 1.0.
    spacing : float, optional
        The amount of space between tetragons. Default is 1.5.
    fig_size : tuple, optional
        Size of the figure (width, height) in inches. Default is (10, 10).
    labels : list, optional
        Labels to be displayed inside each tetragon. Must match the length of x and y.
    label_type : str, optional
        Type of labels to be displayed. Options are:
        - 'manual_labels': Use the provided labels.
        - 'xy': Display the coordinates (x, y) as labels.
        Default is 'column_id'.
    text_size : int, optional
        Font size of the labels inside the tetragons. Default is 10.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    tetragons : list of matplotlib.patches.Polygon
        List of tetragon patch objects used in the plot.
    """
    fig = None  # Initialize fig as None to avoid unbound error
    if ax is None:  # Create new fig and ax if not provided
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')
    
    tetragons = []
    
    def tetragon_vertices(x_center, y_center):
        """Calculate the vertices of a tetragon given its center coordinates."""
        half_size = tet_size / 2
        vertices = [
            (x_center - half_size, y_center - half_size),  # Bottom left
            (x_center + half_size, y_center - half_size),  # Bottom right
            (x_center + half_size, y_center + half_size),  # Top right
            (x_center - half_size, y_center + half_size)   # Top left
        ]
        return vertices
    
    for i in range(len(x)):
        vertices = tetragon_vertices(x[i], y[i])
        tetragon = Polygon(vertices, edgecolor='lightgray', linewidth=0.25, facecolor='none')
        ax.add_patch(tetragon)
        tetragons.append(tetragon)
        
        if label_type == 'manual_labels' and labels:
            label = labels[i]
        elif label_type == 'xy':
            label = f'({x[i]}, {y[i]})'
        else:
            label = ''
        
        if label:
            x_center = x[i]
            y_center = y[i]
            ax.text(x_center, y_center, label, ha='center', va='center', fontsize=text_size)
    
    ax.set_xlim(min(x) - tet_size - spacing, max(x) + tet_size + spacing)
    ax.set_ylim(min(y) - tet_size - spacing, max(y) + tet_size + spacing)
    
    ax.autoscale_view()

    # Remove the box (spines) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    return fig, ax, tetragons

def plot_hex_grid(x, y, hex_size=1.0, spacing=1.5, fig_size=(10, 10), labels=None, label_type='column_id', text_size=10, ax=None):
    """
    Plots a hexagonal grid using the provided x and y coordinates.

    Parameters:
    -----------
    x : list or array-like
        List of x coordinates for the centers of the hexagons.
    y : list or array-like
        List of y coordinates for the centers of the hexagons.
    hex_size : float, optional
        The size of each hexagon, defined as the distance from the center to any vertex. Default is 1.0.
    spacing : float, optional
        The amount of space between hexagons. Default is 1.5.
    fig_size : tuple, optional
        Size of the figure (width, height) in inches. Default is (10, 10).
    labels : list, optional
        Labels to be displayed inside each hexagon. Must match the length of x and y.
    label_type : str, optional
        Type of labels to be displayed. Options are:
        - 'manual_labels': Use the provided labels.
        - 'xy': Display the coordinates (x, y) as labels.
        Default is 'column_id'.
    text_size : int, optional
        Font size of the labels inside the hexagons. Default is 10.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    hexagons : list of matplotlib.patches.Polygon
        List of hexagon patch objects used in the plot.
    """
    fig = None  # Initialize fig as None to avoid unbound error
    if ax is None:  # Create new fig and ax if not provided
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')
    #fig, ax = plt.subplots(figsize=fig_size)
    #ax.set_aspect('equal')
    
    hexagons = []
    
    def hexagon_vertices(x_center, y_center):
        """Calculate the vertices of a hexagon given its center coordinates."""
        angles = np.linspace(0, 2*np.pi, 7) + np.pi/2
        vertices = [(x_center + hex_size * np.cos(angle), y_center + hex_size * np.sin(angle)) for angle in angles]
        return vertices
    
    for i in range(len(x)):
        vertices = hexagon_vertices(x[i], y[i])
        hexagon = Polygon(vertices, edgecolor='lightgray', linewidth=0.25, facecolor='none')
        ax.add_patch(hexagon)
        hexagons.append(hexagon)
        
        if label_type == 'manual_labels' and labels:
            label = labels[i]
        elif label_type == 'xy':
            label = f'({x[i]}, {y[i]})'
        else:
            label = ''
        
        if label:
            x_center = x[i]
            y_center = y[i]
            ax.text(x_center, y_center, label, ha='center', va='center', fontsize=text_size)
    
    ax.set_xlim(min(x) - hex_size - spacing, max(x) + hex_size + spacing)
    ax.set_ylim(min(y) - hex_size - spacing, max(y) + hex_size + spacing)
    
    ax.autoscale_view()

    # Remove the box (spines) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    
    return fig, ax, hexagons


def draw_vector(ax, x_start, y_start, x_end, y_end, linewidth=0.5, head_size=0.25, **kwargs):
    """
    Draws a vector (arrow) on the provided axes from a starting point to an ending point.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes on which to draw the vector.
    x_start : float
        The x coordinate of the start point of the vector.
    y_start : float
        The y coordinate of the start point of the vector.
    x_end : float
        The x coordinate of the end point of the vector.
    y_end : float
        The y coordinate of the end point of the vector.
    linewidth : float, optional
        The width of the vector's line. Default is 2.
    head_size : float, optional
        The size of the arrowhead. Default is 0.5.
    **kwargs : dict, optional
        Additional keyword arguments to customize the arrow properties.

    Returns:
    --------
    None
    """
    arrow_style = f'-|>,head_width={head_size},head_length={head_size * 1}'
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle=arrow_style, linewidth=linewidth, fill=True, **kwargs))



def draw_correlator_vectors(correlator, HR_BL_unique_highest_inputs_filtered, ax, color_dict, to_cell_of_interest):

    # Drawing vectors of a specific correlator
    start_coords_ls = HR_BL_unique_highest_inputs_filtered[f'{correlator}_start_coords']
    end_coords_ls = HR_BL_unique_highest_inputs_filtered[f'{correlator}_end_coords']

    for start_coord, end_coord in zip(start_coords_ls, end_coords_ls):
        # Find the coordinates of the start and end hexagons
        x_start, y_start = start_coord
        x_end, y_end = end_coord

        draw_vector(ax, x_start, y_start, x_end, y_end, color=color_dict[to_cell_of_interest])
        # Check if the start and end coordinates are the same
        if start_coord == end_coord:
            # Plot a single dot at the start_coord
            ax.plot(x_start, y_start, 'o', color=color_dict[to_cell_of_interest],markersize=1)  # 'o' for dot marker
            



def color_hexagons_reference_axes(df_grid, hexagons,parameters_dict,min_max_coordinates_dict,reference_axes_dict,reference_axses_ls, map_type):
    """
    Colors hexagons based on specified reference axes coordinates (p, q, h, v),
    depending on the type of map (regular or hexagonal).

    Parameters:
    hexagons (list): List of hexagon objects to be colored.
    reference_axes_dict (dict): Dictionary containing the reference axes coordinates:
        - 'p_x_ls' (list): List of x-coordinates for the 'p' axis.
        - 'p_y_ls' (list): List of y-coordinates for the 'p' axis.
        - 'q_x_ls' (list): List of x-coordinates for the 'q' axis.
        - 'q_y_ls' (list): List of y-coordinates for the 'q' axis.
        - 'h_x_ls' (list): List of x-coordinates for the horizontal (h) axis (eye's equator).
        - 'h_y_ls' (list): List of y-coordinates for the horizontal (h) axis.
        - 'v_x_ls' (list): List of x-coordinates for the vertical (v) axis (eye's meridian).
        - 'v_y_ls' (list): List of y-coordinates for the vertical (v) axis.
    reference_axses_ls (list): A list containing axis indicators ('p', 'q', 'h', 'v') 
                               that specify which axes' coordinates to use for coloring.
    map_type (str): The type of map being used. Can be 'regular' or 'hexagonal'.

    For each axis in `reference_axses_ls`, the function matches the corresponding coordinates 
    with the positions of the hexagons and colors them accordingly:
    - 'p': Green ('g')
    - 'q': Cyan ('c')
    - 'h': Yellow ('y') for the eye's equator
    - 'v': Grey ('grey') for the eye's meridian

    In 'regular' map mode, the function directly compares the positions of hexagons with the 
    reference coordinates from the provided axes (`p`, `q`, `h`, `v`) to color the hexagons. 
    The horizontal (_180_0_deg_axis) and vertical (_270_90_deg_axis) reference axes are calculated 
    from the provided `h_x_ls`, `h_y_ls`, `v_x_ls`, and `v_y_ls`.

    In 'hexagonal' map mode, the function adjusts the coordinates by adding space between values, 
    recalculates positions using `calculate_new_p_values`, accounts for shifts, and then colors 
    the hexagons accordingly. The new horizontal and vertical reference axes (_180_0_deg_axis 
    and _270_90_deg_axis) are computed based on the new calculated positions.

    Returns:
    _180_0_deg_axis (list of tuples): New or original coordinates for the eye's equator (horizontal axis).
    _270_90_deg_axis (list of tuples): New or original coordinates for the eye's meridian (vertical axis).
    """

    original_p = df_grid.p.tolist()
    original_q = df_grid.q.tolist()

    _space = parameters_dict['space']
    _relative_change = parameters_dict['relative_change']
    center_column_id = df_grid[(df_grid.p == 0) & (df_grid.q == 0)].column_id.values[0] # Centering the original (0,0) coordinate again to 0,0
    center_shift = df_grid[df_grid.column_id == center_column_id].new_p.values[0]
    new_p_values = df_grid['new_p'].tolist() - center_shift

    min_p_value = min_max_coordinates_dict['min_p_value'] 
    max_p_value = min_max_coordinates_dict['max_p_value'] 
    min_q_value = min_max_coordinates_dict['min_q_value'] 
    max_q_value = min_max_coordinates_dict['max_q_value'] 

    h_x_ls = reference_axes_dict['h_x_ls'] 
    h_y_ls = reference_axes_dict['h_y_ls'] 
    v_x_ls = reference_axes_dict['v_x_ls'] 
    v_y_ls = reference_axes_dict['v_y_ls'] 
    p_y_ls = reference_axes_dict['p_y_ls']
    p_x_ls = reference_axes_dict['p_x_ls'] 
    q_x_ls = reference_axes_dict['q_x_ls']
    q_y_ls = reference_axes_dict['q_y_ls']
    h_x_ls_rotated = reference_axes_dict['h_x_ls_rotated'] 
    h_y_ls_rotated = reference_axes_dict['h_y_ls_rotated'] 
    v_x_ls_rotated = reference_axes_dict['v_x_ls_rotated'] 
    v_y_ls_rotated = reference_axes_dict['v_y_ls_rotated'] 

    

    h = list(zip(h_x_ls,h_y_ls))
    v = list(zip(v_x_ls,v_y_ls))
    p = list(zip(p_x_ls,p_y_ls))
    q = list(zip(q_x_ls,q_y_ls))

    if map_type == 'regular':
        for i in reference_axses_ls:
            if i == 'p':
                for p_x, p_y in p:
                    color_in_p = p_x
                    color_in_q = p_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
                        if x_pos == color_in_p and y_pos == color_in_q:
                            hexagon.set_facecolor('lightgreen')
            elif i == 'q':
                for q_x, q_y in q:
                    color_in_p = q_x
                    color_in_q = q_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
                        if x_pos == color_in_p and y_pos == color_in_q:
                            hexagon.set_facecolor('lightblue')
            elif i == 'h':
                # h = eye's equator
                for h_x, h_y in h:
                    color_in_p = h_x
                    color_in_q = h_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
                        if x_pos == color_in_p and y_pos == color_in_q:
                            hexagon.set_facecolor('lightyellow')
            elif i == 'v':
                # v = eye's meridian
                for v_x, v_y in v:
                    color_in_p = v_x
                    color_in_q = v_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(original_p, original_q)):
                        if x_pos == color_in_p and y_pos == color_in_q:
                            hexagon.set_facecolor('lightgray')
        _180_0_deg_axis = list(zip(h_x_ls,h_y_ls)) # This is my reference line to calculate vectors angles. (currently used. It is the eye´s equator)
        _270_90_deg_axis = list(zip(v_x_ls,v_y_ls)) # This is my reference line to calculate vectors angles. (currently NOT used. It is the meridian)

    if map_type == 'regular-rotated':
        for i in reference_axses_ls:
            if i == 'p':
                for hexagon, (x_pos, y_pos) in zip(hexagons, zip(df_grid['rotated_p'], df_grid['rotated_q'])):
                    if x_pos == -y_pos:
                        hexagon.set_facecolor('lightgreen')
            elif i == 'q':
                for hexagon, (x_pos, y_pos) in zip(hexagons, zip(df_grid['rotated_p'], df_grid['rotated_q'])):
                    if x_pos == y_pos:
                        hexagon.set_facecolor('lightblue')
                                    
            elif i == 'h': # h = eye's equator
                for hexagon, (x_pos, y_pos) in zip(hexagons, zip(df_grid['rotated_p'], df_grid['rotated_q'])):
                    if y_pos == 0:
                        hexagon.set_facecolor('lightyellow')
            elif i == 'v': # v = eye's meridian
                for hexagon, (x_pos, y_pos) in zip(hexagons, zip(df_grid['rotated_p'], df_grid['rotated_q'])):
                    if x_pos == 0:
                        hexagon.set_facecolor('lightgray')
                        
        _180_0_deg_axis = list(zip(h_x_ls_rotated,h_y_ls_rotated)) # This is my reference line to calculate vectors angles. (currently used. It is the eye´s equator)
        _270_90_deg_axis = list(zip(v_x_ls_rotated,v_y_ls_rotated)) # This is my reference line to calculate vectors angles. (currently NOT used. It is the meridian
    

    elif map_type == 'hexagonal':
        for i in reference_axses_ls:
            if i == 'p':
                new_p_x_ls = []
                for p_x, p_y in p:
                    color_in_p = p_x
                    color_in_p = add_space_in_between(min_max_coordinates_dict,parameters_dict, [color_in_p])
                    color_in_p = color_in_p[0]
                    color_in_q = p_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_p_values, original_q)):
                        new_p_pos = calculate_new_p_values([color_in_p], [y_pos], start_key=min_q_value , end_key=max_q_value, relative_change=_relative_change)  - center_shift# dealing with shifts
                        if x_pos == new_p_pos[0] and y_pos == color_in_q:
                            hexagon.set_facecolor('lightgreen')
                            new_p_x_ls.append(new_p_pos[0])
            
            elif i == 'q':            
                new_q_x_ls = []
                for q_x, q_y in q:
                    color_in_p = q_x
                    color_in_p = add_space_in_between(min_max_coordinates_dict,parameters_dict, [color_in_p])
                    color_in_p = color_in_p[0]
                    color_in_q = q_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_p_values, original_q)):
                        new_p_pos = calculate_new_p_values([color_in_p], [y_pos], start_key=min_q_value , end_key=max_q_value, relative_change=_relative_change)  - center_shift# dealing with shifts
                        if x_pos == new_p_pos[0] and y_pos == color_in_q:
                            hexagon.set_facecolor('lightblue')
                            new_q_x_ls.append(new_p_pos[0])

            elif i == 'h':                
                new_h_x_ls = []
                for h_x, h_y in h:
                    color_in_p = h_x
                    color_in_p = add_space_in_between(min_max_coordinates_dict,parameters_dict, [color_in_p])
                    color_in_p = color_in_p[0]
                    color_in_q = h_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_p_values, original_q)):
                        new_p_pos = calculate_new_p_values([color_in_p], [y_pos], start_key=min_q_value , end_key=max_q_value, relative_change=_relative_change)  - center_shift# dealing with shifts
                        if x_pos == new_p_pos[0] and y_pos == color_in_q:
                            hexagon.set_facecolor('lightyellow')
                            new_h_x_ls.append(new_p_pos[0])
            
            elif i == 'v':
                new_v_x_ls = []
                for v_x, v_y in v:
                    color_in_p = v_x
                    color_in_p = add_space_in_between(min_max_coordinates_dict,parameters_dict, [color_in_p])
                    color_in_p = color_in_p[0]
                    color_in_q = v_y
                    for hexagon, (x_pos, y_pos) in zip(hexagons, zip(new_p_values, original_q)):
                        new_p_pos = calculate_new_p_values([color_in_p], [y_pos], start_key=min_q_value , end_key=max_q_value, relative_change=_relative_change) - center_shift # dealing with shifts
                        if x_pos == new_p_pos[0] and y_pos == color_in_q:
                            hexagon.set_facecolor('lightgray')
                    new_v_x_ls.append(new_p_pos[0])

        _180_0_deg_axis = list(zip(new_h_x_ls,h_y_ls)) # This is my reference line to calculate vectors angles. (currently used. It is the eye´s equator)
        _270_90_deg_axis = list(zip(new_v_x_ls,v_y_ls)) # This is my reference line to calculate vectors angles. (currently NOT used. It is the meridina)

    return  _180_0_deg_axis, _270_90_deg_axis


def color_polygons_reference_equators(df_grid, polygons, reference_axes_dict, map_type, x_start=-30, x_steps=50):
    """
    Add Docstring once the function is complete
    """
    import matplotlib.colors as mcolors
    # Needed variables
    original_p = df_grid.p.tolist()
    original_q = df_grid.q.tolist()
    h_x_ls = reference_axes_dict['h_x_ls']
    h_x_ls = [x + x_start for x in h_x_ls]
    h_y_ls = reference_axes_dict['h_y_ls']
    
    # Create a yellow color palette with 60 shades
    cmap = plt.get_cmap('YlOrBr', x_steps)  # 'YlOrBr' is a colormap that goes from yellow to brown
    colors = [mcolors.to_hex(cmap(i)) for i in range(x_steps)]  # Generate hex colors for each step
    
    for i in range(x_steps):
        h_x_ls = [x + 1 for x in h_x_ls]
        # Coloring the hexagons
        h = list(zip(h_x_ls, h_y_ls))
        
        if map_type == 'regular':
            # h = eye's equator
            color = colors[i]  # Pick the color for this iteration
            
            for h_x, h_y in h:
                color_in_p = h_x
                color_in_q = h_y
                for polygon, (x_pos, y_pos) in zip(polygons, zip(original_p, original_q)):
                    if x_pos == color_in_p and y_pos == color_in_q:
                        polygon.set_facecolor(color)  # Apply the color from the palette


def plot_nearest_neighbors(df, chosen_column_id):
    import plotly.graph_objects as go
    
    """
    Plot the centroid_x, centroid_y, centroid_z of the chosen row in red,
    the nearest neighbors in blue, equator match IDs in yellow, and all other points in black using Plotly.
    Display column_id when hovering over points.
    """
    # Find the row corresponding to the chosen column_id
    chosen_row = df[df['column_id'] == chosen_column_id]
    
    if chosen_row.empty:
        print(f"Column ID {chosen_column_id} not found in the DataFrame.")
        return
    
    # Extract the coordinates of the chosen row
    chosen_x = chosen_row['centroid_x'].values[0]
    chosen_y = chosen_row['centroid_y'].values[0]
    chosen_z = chosen_row['centroid_z'].values[0]
    
    # Extract the nearest neighbors' column_ids
    nearest_neighbors_ids = chosen_row['nearest_neighbours'].values[0]
    
    # Extract the coordinates and column_ids of the nearest neighbors
    nearest_neighbors = df[df['column_id'].isin(nearest_neighbors_ids)]
    nearest_x = nearest_neighbors['centroid_x'].values
    nearest_y = nearest_neighbors['centroid_y'].values
    nearest_z = nearest_neighbors['centroid_z'].values
    nearest_text = nearest_neighbors['column_id'].values  # Hover text for nearest neighbors
    
    # Extract equator match IDs
    equator_match_ids = chosen_row['equator_match_ids'].values[0]
    
    # Extract the coordinates and column_ids of the equator match IDs
    equator_matches = df[df['column_id'].isin(equator_match_ids)]
    equator_x = equator_matches['centroid_x'].values
    equator_y = equator_matches['centroid_y'].values
    equator_z = equator_matches['centroid_z'].values
    equator_text = equator_matches['column_id'].values  # Hover text for equator match points
    
    # Extract the coordinates and column_ids of all other points
    all_other_neighbors = df[~df['column_id'].isin(nearest_neighbors_ids) & 
                             ~df['column_id'].isin(equator_match_ids) & 
                             (df['column_id'] != chosen_column_id)]
    other_x = all_other_neighbors['centroid_x'].values
    other_y = all_other_neighbors['centroid_y'].values
    other_z = all_other_neighbors['centroid_z'].values
    other_text = all_other_neighbors['column_id'].values  # Hover text for other points
    
    # Create a 3D plot using Plotly
    fig = go.Figure()

    # Add the chosen row in red
    fig.add_trace(go.Scatter3d(
        x=[chosen_x],
        y=[chosen_y],
        z=[chosen_z],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Chosen Row',
        text=[chosen_column_id],  # Hover text for the chosen row
        hoverinfo='text'
    ))

    # Add the nearest neighbors in blue
    fig.add_trace(go.Scatter3d(
        x=nearest_x,
        y=nearest_y,
        z=nearest_z,
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Nearest Neighbors',
        text=nearest_text,  # Hover text for nearest neighbors
        hoverinfo='text'
    ))

    # Add equator match IDs in yellow
    fig.add_trace(go.Scatter3d(
        x=equator_x,
        y=equator_y,
        z=equator_z,
        mode='markers',
        marker=dict(size=6, color='yellow'),
        name='Equator Match IDs',
        text=equator_text,  # Hover text for equator match points
        hoverinfo='text'
    ))

    # Add all other points in black
    fig.add_trace(go.Scatter3d(
        x=other_x,
        y=other_y,
        z=other_z,
        mode='markers',
        marker=dict(size=4, color='black'),
        name='Other Points',
        text=other_text,  # Hover text for other points
        hoverinfo='text'
    ))

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='Centroid X',
            yaxis_title='Centroid Y',
            zaxis_title='Centroid Z'
        ),
        title=f"3D Plot for Column ID {chosen_column_id} and its Nearest Neighbors"
    )

    # Set axis off
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    # Show the plot
    fig.show()
    return fig

def plot_nearest_neighbors_and_vector(df, chosen_column_id):
    import plotly.graph_objects as go
    
    """
    Plot the centroid_x, centroid_y, centroid_z of the chosen row in red,
    the nearest neighbors in blue, equator match IDs in yellow, two green points (vector start end end position), and all other points in black using Plotly.
    Display column_id when hovering over points.
    """
    # Find the row corresponding to the chosen column_id
    chosen_row = df[df['column_id'] == chosen_column_id]
    
    if chosen_row.empty:
        print(f"Column ID {chosen_column_id} not found in the DataFrame.")
        return
    
    # Extract the coordinates of the chosen row
    chosen_x = chosen_row['centroid_x'].values[0]
    chosen_y = chosen_row['centroid_y'].values[0]
    chosen_z = chosen_row['centroid_z'].values[0]
    
    # Extract the nearest neighbors' column_ids
    nearest_neighbors_ids = chosen_row['nearest_neighbours'].values[0]
    
    # Extract the coordinates and column_ids of the nearest neighbors
    nearest_neighbors = df[df['column_id'].isin(nearest_neighbors_ids)]
    nearest_x = nearest_neighbors['centroid_x'].values
    nearest_y = nearest_neighbors['centroid_y'].values
    nearest_z = nearest_neighbors['centroid_z'].values
    nearest_text = nearest_neighbors['column_id'].values  # Hover text for nearest neighbors
    
    # Extract equator match IDs
    equator_match_ids = chosen_row['equator_match_ids'].values[0]
    
    # Extract the coordinates and column_ids of the equator match IDs
    equator_matches = df[df['column_id'].isin(equator_match_ids)]
    equator_x = equator_matches['centroid_x'].values
    equator_y = equator_matches['centroid_y'].values
    equator_z = equator_matches['centroid_z'].values
    equator_text = equator_matches['column_id'].values  # Hover text for equator match points
    
    # Extract the coordinates and column_ids of all other points
    all_other_neighbors = df[~df['column_id'].isin(nearest_neighbors_ids) & 
                             ~df['column_id'].isin(equator_match_ids) & 
                             (df['column_id'] != chosen_column_id)]
    other_x = all_other_neighbors['centroid_x'].values
    other_y = all_other_neighbors['centroid_y'].values
    other_z = all_other_neighbors['centroid_z'].values
    other_text = all_other_neighbors['column_id'].values  # Hover text for other points
    
    # Extract coordinates for the two additional points
    mi1_x = chosen_row['center_mi1_x'].values[0]
    mi1_y = chosen_row['center_mi1_y'].values[0]
    mi1_z = chosen_row['center_mi1_z'].values[0]
    
    mi4_x = chosen_row['center_mi4_x'].values[0]
    mi4_y = chosen_row['center_mi4_y'].values[0]
    mi4_z = chosen_row['center_mi4_z'].values[0]

    # Get root_ids
    mi1_root_id = chosen_row['Mi1_ID'].values[0]
    mi4_root_id = chosen_row['Mi4_ID'].values[0]
    
    
    # Create a 3D plot using Plotly
    fig = go.Figure()

    # Add the chosen row in red
    fig.add_trace(go.Scatter3d(
        x=[chosen_x],
        y=[chosen_y],
        z=[chosen_z],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Chosen Row',
        text=[chosen_column_id],  # Hover text for the chosen row
        hoverinfo='text'
    ))

    # Add the nearest neighbors in blue
    fig.add_trace(go.Scatter3d(
        x=nearest_x,
        y=nearest_y,
        z=nearest_z,
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Nearest Neighbors',
        text=nearest_text,  # Hover text for nearest neighbors
        hoverinfo='text'
    ))

    # Add equator match IDs in yellow
    fig.add_trace(go.Scatter3d(
        x=equator_x,
        y=equator_y,
        z=equator_z,
        mode='markers',
        marker=dict(size=6, color='yellow'),
        name='Equator Match IDs',
        text=equator_text,  # Hover text for equator match points
        hoverinfo='text'
    ))

    # Add all other points in black
    fig.add_trace(go.Scatter3d(
        x=other_x,
        y=other_y,
        z=other_z,
        mode='markers',
        marker=dict(size=4, color='black'),
        name='Other Points',
        text=other_text,  # Hover text for other points
        hoverinfo='text'
    ))
    
    # Add the two green points (center_mi1 and center_mi4)
    fig.add_trace(go.Scatter3d(
        x=[mi1_x, mi4_x],
        y=[mi1_y, mi4_y],
        z=[mi1_z, mi4_z],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Vector Points',
        text=[f'Mi1: {mi1_root_id}', f'Mi4: {mi4_root_id}'],  # Hover text for the green points
        hoverinfo='text'
    ))

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='Centroid X',
            yaxis_title='Centroid Y',
            zaxis_title='Centroid Z'
        ),
        title=f"3D Plot for Column ID {chosen_column_id} and its Nearest Neighbors"
    )

    # Set axis off
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    # Show the plot
    fig.show()
    return fig



def plot_local_plane_coords(df, chosen_column_id):
    import plotly.graph_objects as go
    """
    Plot the local_plane_coords for a chosen column_id with the centroid of the chosen row in red,
    and the local_plane_coords of its nearest neighbors in blue.

    Parameters:
    - df: DataFrame containing the data with the local_plane_coords column.
    - chosen_column_id: The column_id of the chosen row.
    """
    # Find the row corresponding to the chosen column_id
    chosen_row = df[df['column_id'] == chosen_column_id]
    
    if chosen_row.empty:
        print(f"Column ID {chosen_column_id} not found in the DataFrame.")
        return
    
    # Extract the local plane coordinates of the chosen row
    chosen_local_coords = chosen_row['local_plane_coords'].values[0]
    
    # Extract the nearest neighbors' column_ids
    nearest_neighbors_ids = chosen_row['nearest_neighbours'].values[0]
    
    # Extract the local plane coordinates of the nearest neighbors
    nearest_neighbors = df[df['column_id'].isin(nearest_neighbors_ids)]
    nearest_local_coords = nearest_neighbors['local_plane_coords'].values
    
    # Create a list to hold all nearest neighbor coordinates
    all_nearest_coords = []
    for coords in nearest_local_coords:
        all_nearest_coords.extend(coords)  # Flatten the list of lists

    # Prepare data for plotting
    chosen_x = [coord[0] for coord in chosen_local_coords]
    chosen_y = [coord[1] for coord in chosen_local_coords]
    
    nearest_x = [coord[0] for coord in all_nearest_coords]
    nearest_y = [coord[1] for coord in all_nearest_coords]
    
    # Create a scatter plot using Plotly
    fig = go.Figure()

    # Plot the chosen row in red
    fig.add_trace(go.Scatter(
        x=chosen_x,
        y=chosen_y,
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Chosen Row'
    ))

    # Plot the nearest neighbors in blue
    fig.add_trace(go.Scatter(
        x=nearest_x,
        y=nearest_y,
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Nearest Neighbors'
    ))

    # Set labels and title
    fig.update_layout(
        xaxis_title='PC1',
        yaxis_title='PC2',
        title=f"2D Projection of Local Plane for Column ID {chosen_column_id}"
    )

    # Show the plot
    fig.show()


