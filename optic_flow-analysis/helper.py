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

#%% Plotting functions

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




