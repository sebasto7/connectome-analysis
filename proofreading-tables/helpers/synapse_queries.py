import pandas as pd

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




def separate_xyz(df):
    """
    Separates [x,y,z] lists of pre_pt_position and post_pt_position into separeted columns

    Args:
        pandas data frame containig [x,y,z] lists under pre_pt_root_id and post_pt_root_id column names

    Returns:
        same pandas data frame containing x,y and z columns of pre- and postsynapses of the same length
    """
    curr_list_arr = df['pre_pt_position'].tolist()
    curr_list_tuple = list(map(tuple,curr_list_arr)) #From list of arrays to list of tuples
    curr_list_list = list(map(list, zip(*curr_list_tuple))) #From list of tuples to individual lists
    df['pre_x'], df['pre_y'], df['pre_z'] = [curr_list_list[0], curr_list_list[1], curr_list_list[2]] # Adding the new columns

    curr_list_arr = df['post_pt_position'].tolist()
    curr_list_tuple = list(map(tuple,curr_list_arr)) #From list of arrays to list of tuples
    curr_list_list = list(map(list, zip(*curr_list_tuple))) #From list of tuples to individual lists
    df['post_x'], df['post_y'], df['post_z'] = [curr_list_list[0], curr_list_list[1], curr_list_list[2]] # Adding the new columns




def synapse_count(pre_df, post_df):
    """
    Counting inputs and ouputs per ID

    Args:
        'pre_df' and 'post_df': two dataframes containing 'pre_pt_root_id' and 'post_pt_root_id' as columns

    Retuns:
        'count_pre_str_df' and 'count_post_str_df': two dataframes with counts of pre and postsynapse respectively
        with every synaptic partner.
        'total_synapse_count_df': single data frame contaning the sum of pre and postsynapti sites per each ID

    """

    count_pre_df = pd.DataFrame() # This will contain the synapse count with each partner
    count_post_df = pd.DataFrame()# This will contain the synapse count with each partner

    pre_IDs = pre_df['pre_pt_root_id'].unique()
    post_IDs = post_df['post_pt_root_id'].unique()

    # 1. Looping through the two data frames

    # For pre_df
    for n in pre_df['pre_pt_root_id'].unique():
        pre_count = {}
        curr_pre = pre_df[pre_df['pre_pt_root_id'] == n]
        pre_str = curr_pre.applymap(str)

        for c in pre_str['post_pt_root_id'].to_list():
            pre_count[c] = pre_count.get(c, 0) + 1
        pre_count_df = pd.DataFrame(pre_count, index=[0])
        pre_count_df = pre_count_df.T
        pre_count_df.rename(columns={0: "counts"},inplace=True)
        pre_count_df.index.names = ['postsynaptic_ID']
        pre_count_df = pre_count_df.sort_values(by="counts",ascending=False)
        pre_count_df['presynaptic_ID'] = pre_str['pre_pt_root_id'].to_list()[0:len(pre_count_df)]
        count_pre_df = count_pre_df.append(pre_count_df)

    count_pre_str_df = count_pre_df.applymap(str)

    # For post_df
    for n in post_df['post_pt_root_id'].unique():
        post_count = {}
        curr_post = post_df[post_df['post_pt_root_id'] == n]
        post_str = curr_post.applymap(str)

        for c in post_str['pre_pt_root_id'].to_list():
            post_count[c] = post_count.get(c, 0) + 1
        post_count_df = pd.DataFrame(post_count, index=[0])
        post_count_df = post_count_df.T
        post_count_df.rename(columns={0: "counts"},inplace=True)
        post_count_df.index.names = ['presynaptic_ID']
        post_count_df = post_count_df.sort_values(by="counts",ascending=False)
        post_count_df['postsynaptic_ID'] = post_str['post_pt_root_id'].to_list()[0:len(post_count_df)]
        count_post_df = count_post_df.append(post_count_df)

    count_post_str_df = count_post_df.applymap(str)


    # 2.  Getting total counts in a single data frame

    total_count_pre = count_pre_df.groupby(['presynaptic_ID']).counts.sum().reset_index()
    total_count_pre.rename(columns={"counts": "pre","presynaptic_ID": "id"},inplace=True)

    total_count_post = count_post_df.groupby(['postsynaptic_ID']).counts.sum().reset_index()
    total_count_post.rename(columns={"counts": "post","postsynaptic_ID": "id" },inplace=True)

    total_synapse_count_df = pd.merge(total_count_pre, total_count_post, on='id')
    total_synapse_count_df.set_index('id', inplace=True)

    return count_pre_str_df, count_post_str_df, total_synapse_count_df


import math

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in three-dimensional space.

    Args:
        point1 (list): The coordinates of the first point in XYZ format.
        point2 (list): The coordinates of the second point in XYZ format.

    Returns:
        float: The Euclidean distance between the two points in nanometers.
    """
    
    resolution_x = 4  # nm
    resolution_y = 4  # nm
    resolution_z = 40  # nm

    # Scale the coordinates using the resolution values
    scaled_point1 = [point1[0] * resolution_x, point1[1] * resolution_y, point1[2] * resolution_z]
    scaled_point2 = [point2[0] * resolution_x, point2[1] * resolution_y, point2[2] * resolution_z]

    # Step 1: Calculate distance in X and Y coordinates separately
    diff_x = scaled_point2[0] - scaled_point1[0]
    diff_y = scaled_point2[1] - scaled_point1[1]
    distance_x = abs(diff_x)
    distance_y = abs(diff_y)

    # Step 2: Calculate the sum of distances in X and Y coordinates
    distance_xy = math.sqrt(distance_x**2 + distance_y**2)

    # Step 3: Assign distance_xy to variable "a"
    a = distance_xy

    # Step 4: Calculate distance in Z coordinate
    diff_z = scaled_point2[2] - scaled_point1[2]
    distance_z = abs(diff_z)

    # Step 5: Assign distance_z to variable "b"
    b = distance_z

    # Step 6: Calculate the hypotenuse "h" using Pythagoras' theorem
    h = math.sqrt(a**2 + b**2)

    return h


def filter_points(points, threshold_distance):
    """
    Filter out points that are closer than the given threshold distance to any other point in the list.

    Args:
        points (list): List of points in XYZ format.
        threshold_distance (float): Threshold distance in nanometers.

    Returns:
        list: Filtered list of points that are not closer than the threshold distance to any other point.
    """
    filtered_points = []
    num_points = len(points)

    for i in range(num_points):
        point_i = points[i]
        discard_point = False

        for j in range(i + 1, num_points):
            point_j = points[j]
            distance = calculate_distance(point_i, point_j)

            if distance < threshold_distance:
                discard_point = True
                break

        if not discard_point:
            filtered_points.append(point_i)

    return filtered_points

