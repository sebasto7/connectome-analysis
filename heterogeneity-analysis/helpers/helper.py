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


def update_dataframe_single_column(source_df, target_df, reference_column):
    """
    Update the rows of the target DataFrame with corresponding rows from the source DataFrame
    based on a common reference column.

    Args:
        source_df (pd.DataFrame): The DataFrame containing the updated data.
        target_df (pd.DataFrame): The DataFrame to be updated.
        reference_column (str): The name of the column to be used as the reference for matching rows.

    Returns:
        pd.DataFrame: The updated target DataFrame where rows have been replaced based on the reference column.
    """
    # Create a dictionary mapping reference column values to corresponding rows in the source DataFrame
    reference_dict = source_df.groupby(reference_column).first().reset_index().to_dict(orient='records')
    reference_dict = {row[reference_column]: row for row in reference_dict}

    # Update the target DataFrame based on the reference column
    for i, row in target_df.iterrows():
        ref = row[reference_column]
        if ref in reference_dict:
            source_row = reference_dict[ref]
            target_df.loc[i] = source_row

    return target_df

    def update_dataframe(source_df, target_df, reference_column1, reference_column2):
    """
    Update the rows of the target DataFrame with corresponding rows from the source DataFrame
    based on two reference columns.

    Args:
        source_df (pd.DataFrame): The DataFrame containing the updated data.
        target_df (pd.DataFrame): The DataFrame to be updated.
        reference_column1 (str): The name of the first column used as the reference for matching rows.
        reference_column2 (str): The name of the second column used as the reference for matching rows.

    Returns:
        pd.DataFrame: The updated target DataFrame where rows have been replaced based on the reference columns.
    """
    # Create a dictionary mapping pairs of reference column values to corresponding rows in the source DataFrame
    reference_columns = [reference_column1, reference_column2]
    reference_dict = source_df.groupby(reference_columns).first().reset_index().to_dict(orient='records')
    reference_dict = {(row[reference_column1], row[reference_column2]): row for row in reference_dict}

    # Update the target DataFrame based on the reference columns
    for i, row in target_df.iterrows():
        ref1 = row[reference_column1]
        ref2 = row[reference_column2]
        if (ref1, ref2) in reference_dict:
            source_row = reference_dict[(ref1, ref2)]
            target_df.loc[i] = source_row

    return target_df


    def find_center_point(points, threshold):
    """
    Find the geometric center point of a subset of points that are within a given average distance threshold,
    and determine the closest point to this center.

    Args:
        points (list or np.ndarray): A list or NumPy array of shape (n, 3), where each entry is a point in 3D space.
        threshold (float): The maximum average distance from which points are considered valid for calculating the center.

    Returns:
        tuple: A tuple containing:
            - center_point (list): The geometric center point of the valid subset of points, rounded to one decimal place.
            - closest_point (list): The point from the valid subset that is closest to the computed center point.
    """
    if isinstance(points, list):
        points = np.array(points)

    # Calculate the distances between each point and all other points
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    # Calculate the average distance for each point
    avg_distances = np.mean(distances, axis=1)

    # Find the indices of points within the threshold distance
    valid_indices = np.where(avg_distances < threshold)[0]

    # Check if there are any valid points
    if len(valid_indices) > 0:
        # Calculate the geometric center of valid points
        center_point = np.mean(points[valid_indices], axis=0)
        # Round the center point to one decimal place
        center_point = np.round(center_point, decimals=1)
        
        # Find the closest point to the center
        closest_point_index = np.argmin(np.linalg.norm(points[valid_indices] - center_point, axis=1))
        closest_point = points[valid_indices][closest_point_index]
    else:
        # Default values when no points are within the threshold
        center_point = np.array([0, 0, 0])
        closest_point = np.array([0, 0, 0])

    return center_point.tolist(), closest_point.tolist()


def save_list_to_file(file_path, input_list):
    """
    Save a list of items to a CSV file, with each item on a new line.

    Args:
        file_path (str): The path to the file where the list will be saved. If the file already exists, it will be overwritten.
        input_list (list): A list of items to be saved to the CSV file.

    Returns:
        None
    """
    df = pd.DataFrame(input_list, columns=['Items'])
    df.to_csv(file_path, header=False, index=False)






