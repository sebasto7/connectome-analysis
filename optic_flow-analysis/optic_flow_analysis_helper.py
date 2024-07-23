## Custom functions for creating the 2d lattice

"""
Created on Mon July 22 11:35:00 2024

@author: smolina

helper for the optic flow analysis of DS cells in FAFB data set
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def plot_hex_grid(x, y, hex_size=1.0, spacing=1.5, fig_size=(10, 10), labels=None, label_type='column_id', text_size=10):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    
    hexagons = []
    
    # Function to calculate vertices of hexagon
    def hexagon_vertices(x_center, y_center):
        angles = np.linspace(0, 2*np.pi, 7) + np.pi/2  # Add pi/8 to rotate by 45 degrees
        vertices = [(x_center + hex_size * np.cos(angle), y_center + hex_size * np.sin(angle)) for angle in angles]
        return vertices
    
    # Plot hexagons
    for i in range(len(x)):
        vertices = hexagon_vertices(x[i], y[i])
        hexagon = Polygon(vertices, edgecolor='black', linewidth=1, facecolor='none')
        ax.add_patch(hexagon)
        hexagons.append(hexagon)
        
        # Determine label based on label_type
        if label_type == 'manual_labels' and labels:
            label = labels[i]
        elif label_type == 'xy':
            label = f'({x[i]}, {y[i]})'
        else:
            label = ''
        
        # Add label in the center of the hexagon
        if label:
            x_center = x[i]
            y_center = y[i]
            ax.text(x_center, y_center, label, ha='center', va='center', fontsize=text_size)
    
    # Set limits based on the hexagon positions
    ax.set_xlim(min(x) - hex_size - spacing, max(x) + hex_size + spacing)
    ax.set_ylim(min(y) - hex_size - spacing, max(y) + hex_size + spacing)
    
    ax.autoscale_view()
    
    return fig, ax, hexagons

def draw_vector(ax, x_start, y_start, x_end, y_end, linewidth=2, head_size=0.5, **kwargs):
    """Draw an arrow (vector) from one (x, y) coordinate to another with specified linewidth and arrowhead size."""
    arrow_style = f'->,head_width={head_size},head_length={head_size * 1.5}'
    arrow_style = f'-|>,head_width={head_size},head_length={head_size * 1.5}'
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle=arrow_style, linewidth=linewidth, fill=True, **kwargs))


def calculate_new_p_values(original_x, original_y, start_key=-16, end_key=17, relative_change=0.5, in_space=0):
    shift_dict = {}

    # Iterate through the range from start_key to end_key (inclusive)
    for i in range(start_key, end_key + 1):
        # Calculate the corresponding value based on relative_change
        value = (i - start_key) * relative_change
        # Assign the key-value pair to the dictionary
        shift_dict[i] = value

    # Initialize a list to store the new_x values
    new_x_values = []

    # Iterate through each pair of original_x and original_y
    for x, y in zip(original_x, original_y):
        if y in shift_dict:
            # Get the corresponding relative_change from shift_dict
            relative_change = shift_dict[y]
            # Calculate the new_x value
            new_x = x - relative_change
            # Append the new_x value to the list
            new_x_values.append(new_x)
        else:
            # Handle case where y is not found in shift_dict (optional)
            # You may choose to skip these or handle them differently
            new_x_values.append(x)  # Keep original x value if no match found


    return new_x_values


def add_space_in_between(num_negatives, num_positives, space, key_list):
    # Generate negative numbers
    negative_numbers = [-space * i for i in range(num_negatives, 0, -1)]
    
    # Generate positive numbers
    positive_numbers = [space * i for i in range(1, num_positives + 1)]
    
    # Combine the lists with 0 in between
    generated_list = negative_numbers + [0] + positive_numbers
    
    # Create the original range list including 0
    original_range = list(range(-num_negatives, num_positives + 1))
    
    # Create a dictionary with original_range as keys and generated_list as values
    space_dict = {}
    
    # Fill the dictionary with corresponding values
    for i, key in enumerate(original_range):
        if i < len(generated_list):
            space_dict[key] = generated_list[i]
        else:
            space_dict[key] = None  # Assign None if there's no corresponding value

    # Generate a new list based on the key_list and space_dict
    new_list = [space_dict.get(key, None) for key in key_list]
    
    return new_list
