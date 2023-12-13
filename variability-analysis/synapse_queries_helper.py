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




def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in three-dimensional space.

    Args:
        point1 (list): The coordinates of the first point in XYZ format.
        point2 (list): The coordinates of the second point in XYZ format.

    Returns:
        float: The Euclidean distance between the two points in nanometers.
    """
    import math
    
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




def calculate_distance_nearest_neighbour(points, lower_threshold, upper_threshold):
    """
    Calculate the distances from each point to its nearest neighbor in a list of points in 3D space.
    Only distances within the specified threshold values are included in the output list.

    Args:
        points (list): A list of points in XYZ format.
        lower_threshold (float): The lower threshold value for distances to be included.
        upper_threshold (float): The upper threshold value for distances to be included.

    Returns:
        list: A list of distances within the specified threshold values.
    """
    
    import math
    
    resolution_x = 4  # nm
    resolution_y = 4  # nm
    resolution_z = 40  # nm

    # Scale the coordinates of all points using the resolution values
    scaled_points = [[point[0] * resolution_x, point[1] * resolution_y, point[2] * resolution_z] for point in points]

    distances = []
    
    for i, point1 in enumerate(scaled_points):
        nearest_distance = math.inf
        
        for j, point2 in enumerate(scaled_points):
            if i != j:  # Skip the same point
                # Step 1: Calculate distance in X and Y coordinates separately
                diff_x = point2[0] - point1[0]
                diff_y = point2[1] - point1[1]
                distance_x = abs(diff_x)
                distance_y = abs(diff_y)

                # Step 2: Calculate the sum of distances in X and Y coordinates
                distance_xy = math.sqrt(distance_x**2 + distance_y**2)

                # Step 3: Assign distance_xy to variable "a"
                a = distance_xy

                # Step 4: Calculate distance in Z coordinate
                diff_z = point2[2] - point1[2]
                distance_z = abs(diff_z)

                # Step 5: Assign distance_z to variable "b"
                b = distance_z

                # Step 6: Calculate the hypotenuse "h" using Pythagoras' theorem
                h = math.sqrt(a**2 + b**2)

                if h < nearest_distance:
                    nearest_distance = h
        
        if lower_threshold <= nearest_distance <= upper_threshold:
            distances.append(nearest_distance)

    return distances


def calculate_neuron_weights(pre_post_counts, post_inputs, up_to_date_pre_ids, min_desired_count):
    # Synaptic counts filter
    pre_post_counts = pre_post_counts[pre_post_counts['pre_syn_count'] >= min_desired_count].copy()

    # Getting all input counts for each presynaptic neuron id
    post_inputs_counts = post_inputs.groupby(['post_pt_root_id', 'pre_pt_root_id'])['pre_pt_root_id'].count().reset_index(name='pre_syn_count')

    # Initializing variables
    rel_neuron_type_weight = []
    rel_neuron_weight = []
    neuron_weight_post_ls = []
    neuron_weight_pre_ls = []

    # Looping to get the neuron and neuron type weights for each postsynaptic partner
    for _post in list(set(post_inputs_counts['post_pt_root_id'].tolist())):
        curr_df = post_inputs_counts[post_inputs_counts['post_pt_root_id'] == _post].copy()
        curr_total_syn_count = curr_df['pre_syn_count'].sum()
        curr_pre_type_df = curr_df[curr_df['pre_pt_root_id'].isin(up_to_date_pre_ids)].copy()

        # Weight for all cells of the same type
        curr_rel_neuron_type_weight = curr_pre_type_df['pre_syn_count'].sum() / curr_total_syn_count
        rel_neuron_type_weight.append(curr_rel_neuron_type_weight)

        # Weight for individual cells of the same type
        if len(curr_pre_type_df['pre_pt_root_id']) == 0:
            rel_neuron_weight.append(0.0)
            neuron_weight_post_ls.append(_post)  # Tracking the postsynaptic neuron ids
            neuron_weight_pre_ls.append('-')  # Tracking the presynaptic neuron ids
        else:
            for _pre in list(set(curr_pre_type_df['pre_pt_root_id'].tolist())):
                neuron_weight_post_ls.append(_post)  # Tracking the postsynaptic neuron ids
                neuron_weight_pre_ls.append(_pre)  # Tracking the presynaptic neuron ids
                curr_rel_neuron_weight = curr_pre_type_df[curr_pre_type_df['pre_pt_root_id'] == _pre]['pre_syn_count'].sum() / curr_total_syn_count
                rel_neuron_weight.append(curr_rel_neuron_weight)

    # Summary data frames
    neuron_weight_df = pd.DataFrame()
    neuron_weight_df['rel_weight'] = rel_neuron_weight
    neuron_weight_df['post'] = neuron_weight_post_ls
    neuron_weight_df['pre'] = neuron_weight_pre_ls

    return neuron_weight_df, rel_neuron_type_weight




def match_all_pre_to_single_post(up_to_date_post_ids, up_to_date_pre_ids, neuropile_mesh):
    
    from fafbseg import flywire
    print('Matching all pre to single post')
    
    # Fetch the neuron's inputs
    post_inputs = flywire.synapses.fetch_synapses(
        up_to_date_post_ids, pre=False, post=True, attach=True,
        min_score=50, clean=True, transmitters=False,
        neuropils=True, batch_size=30,
        dataset='production', progress=True, mat="live"
    )

    # Combining pre- and postsynapses XYZ values in single columns
    combine_xyz(post_inputs)  # Assuming combine_xyz is a defined function that does the operation

    # Filtering: keeping only synapses in the medulla
    post_inputs = post_inputs[post_inputs['neuropil'] == neuropile_mesh].copy()

    # Filter connections just selected presynaptic cells
    pre_post_match_df = post_inputs[post_inputs['pre_pt_root_id'].isin(up_to_date_pre_ids)].copy()

    # Aggregating data frame based on unique post and pre segment IDs
    # While aggregating, counting the number of contacts for each pre-post pair
    pre_post_counts = pre_post_match_df.groupby(['post_pt_root_id', 'pre_pt_root_id'])['pre_pt_root_id'].count().reset_index(name='pre_syn_count')


    return pre_post_counts, post_inputs




def calculate_spatial_span(up_to_date_post_ids, up_to_date_pre_ids, post_ids_update_df, R_post_df, post_inputs, pre_post_counts, pre_inputs, single_column_area,single_column_diameter):
    """
    Calculates the TOTAL spatial span of all presynaptic neurons in the list that together contact the same postsynaptic neuron,
    for a list of postsynaptic neurons
    """
   
    
    import pandas as pd
    import numpy as np
    from scipy.spatial import ConvexHull, distance
    from scipy import stats
    print('Calculating spatial span')
    
    #For all presynaptic neurons that togeter contact same postsynaptic neuron:
    pre_post_volumes = []
    pre_post_areas = []
    pre_post_diameters = []
    pre_post_diameters_projected = []
    pre_count = []
    pre_xzy_ls = []
    post_xzy_ls = []
    pre_center_ls = []
    num_pre_sites = []
    hull_ls = []
    pre_projected_points_ls = []
    
    #For individual presynaptic neurons:
    individual_pre_xzy_ls = []
    individual_post_xzy_ls = []
    individual_pre_center_ls = []
    individual_num_pre_sites = []
    individual_pre_post_volumes = []
    individual_pre_post_areas = []
    individual_pre_post_diameters = []
    individual_pre_post_diameters_projected = []
    individual_hull_ls = []
    individual_pre_count = []
    individual_curr_post = []
    individual_pre_projected_points_ls = []
    

    for i in range(0, len(up_to_date_post_ids)):
        curr_post = up_to_date_post_ids[i]

        # Getting single postynaptic cell's coordinates
        try:
            old_curr_post = update_df[update_df['new_id'] == curr_post]['old_id'].tolist()[0]
        except:
            old_curr_post = str(curr_post)

        single_post_coords = R_post_df[R_post_df['Updated_seg_id'] == old_curr_post]['XYZ-ME'].to_numpy(dtype=str, copy=True)
        post_xyz = np.zeros([np.shape(single_post_coords)[0], 3])
        new_post_coords = np.zeros([np.shape(single_post_coords)[0], 3])

        for idx, coordinate in enumerate(single_post_coords):
            post_xyz[idx, :] = np.array([coordinate.split(',')], dtype=float)
            new_post_coords[idx, :] = np.array([coordinate.split(',')], dtype=float)

        post_xyz *= [4, 4, 40]  # For plotting it using navis (correcting for data resolution)
        post_xzy_ls.append(post_xyz)
        

        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_post_inputs = post_inputs[post_inputs['post_pt_root_id'] == curr_post].copy()

        
        ## For all presynaptic neurons that togeter contact same postsynaptic neuron:
        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_pre_ls = pre_post_counts[pre_post_counts['post_pt_root_id'] == curr_post]['pre_pt_root_id'].tolist()
        curr_pre_inputs = pre_inputs[pre_inputs['post_pt_root_id'].isin(curr_pre_ls)].copy()

        if len(curr_pre_inputs) < 10:
            pre_post_volumes.append(None)
            pre_post_areas.append(None)
            pre_count.append(None)
            pre_xzy_ls.append(None)
            pre_center_ls.append(None)
            num_pre_sites.append(None)
            hull_ls.append(None)
            pre_post_diameters.append(None)
            pre_post_diameters_projected.append(None)
            pre_projected_points_ls.append(None)
        else:
            pre_count.append(len(curr_pre_ls))

            # Getting presynaptic cells coordinates
            temp_pre_coords = curr_pre_inputs['pre_pt_position'].tolist()

            # Correcting xyz positions for mesh plotting
            pre_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_pre_coords])
            pre_xzy_ls.append(pre_xyz)
            num_pre_sites.append(len(pre_xyz))  # Total number of points in the presynaptic partner(s)

            # Calculate the center of the cloud of points
            pre_center = np.mean(pre_xyz, axis=0)
            pre_center_ls.append(pre_center)

            # Calculate the volume of the cloud using the convex hull method
            hull = ConvexHull(pre_xyz)
            volume = hull.volume
            pre_post_volumes.append(volume)
            
            # Calculate largest diameter
            largest_diameter = 0
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        # Calculate distance between two points
                        d = distance.euclidean(pre_xyz[simplex[i]], pre_xyz[simplex[j]])
                        if d > largest_diameter:
                            largest_diameter = d

            # Convert largest diameter to micrometers
            largest_diameter_um = largest_diameter / 10**3
            pre_post_diameters.append(largest_diameter_um)

            # Calculate volume/area based on projections using PCA on presynaptic partner coordinates
            # PCA to get an approximate area of the volume
            pre_mean = np.mean(pre_xyz, axis=0)
            pre_centered_points = pre_xyz - pre_mean
            pre_cov_matrix = np.cov(pre_centered_points, rowvar=False)
            pre_eigenvalues, pre_eigenvectors = np.linalg.eigh(pre_cov_matrix)
            pre_normal_vector = pre_eigenvectors[:, [1, 2]]  # PC2 and PC3

            # Calculate volume/area based on projections using PCA on postsynaptic partner coordinates
            temp_post_coords = curr_post_inputs['pre_pt_position'].tolist()
            post_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_post_coords])
            post_mean = np.mean(post_xyz, axis=0)
            post_centered_points = post_xyz - post_mean
            post_cov_matrix = np.cov(post_centered_points, rowvar=False)
            post_eigenvalues, post_eigenvectors = np.linalg.eigh(post_cov_matrix)
            post_normal_vector = post_eigenvectors[:, [0, 1]]  # PC1 and PC2

            # Project the points
            projected_points = pre_centered_points.dot(post_normal_vector)
            pre_projected_points_ls.append(projected_points)

            # Calculate area
            hull = ConvexHull(projected_points)
            hull_ls.append(hull)
            area = hull.volume  # Area is calculated as volume in 2D
            area_um2 = area / 10**6
            pre_post_areas.append(area_um2)
            
            # Calculate largest diameter
            largest_diameter = 0
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        # Calculate distance between two points
                        d = distance.euclidean(projected_points[simplex[i]], projected_points[simplex[j]])
                        if d > largest_diameter:
                            largest_diameter = d

            # Convert largest diameter to micrometers
            largest_diameter_um = largest_diameter / 10**3
            pre_post_diameters_projected.append(largest_diameter_um)
            
            
        ## For individual presynaptic neurons:
        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_pre_ls = pre_post_counts[pre_post_counts['post_pt_root_id'] == curr_post]['pre_pt_root_id'].tolist()
        for curr_pre in curr_pre_ls:
            individual_post_xzy_ls.append(post_xyz)
            individual_curr_post.append(curr_post)
            curr_pre_inputs = pre_inputs[pre_inputs['post_pt_root_id'].isin([curr_pre])].copy()
            if len(curr_pre_inputs) < 5:
                individual_pre_post_volumes.append(None)
                individual_pre_post_areas.append(None)
                individual_pre_count.append(None)
                individual_pre_xzy_ls.append(None)
                individual_pre_center_ls.append(None)
                individual_num_pre_sites.append(None)
                individual_hull_ls.append(None)
                individual_pre_post_diameters.append(None)
                individual_pre_post_diameters_projected.append(None)
                individual_pre_projected_points_ls.append(None)
            else:
                individual_pre_count.append(len([curr_pre]))

                # Getting presynaptic cells coordinates
                temp_pre_coords = curr_pre_inputs['pre_pt_position'].tolist()

                # Correcting xyz positions for mesh plotting
                pre_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_pre_coords])
                individual_pre_xzy_ls.append(pre_xyz)
                individual_num_pre_sites.append(len(pre_xyz))  # Total number of points in the presynaptic partner(s)

                # Calculate the center of the cloud of points
                pre_center = np.mean(pre_xyz, axis=0)
                individual_pre_center_ls.append(pre_center)

                # Calculate the volume of the cloud using the convex hull method
                hull = ConvexHull(pre_xyz)
                volume = hull.volume
                individual_pre_post_volumes.append(volume)
                
                # Calculate largest diameter
                largest_diameter = 0
                for simplex in hull.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            # Calculate distance between two points
                            d = distance.euclidean(pre_xyz[simplex[i]], pre_xyz[simplex[j]])
                            if d > largest_diameter:
                                largest_diameter = d

                # Convert largest diameter to micrometers
                largest_diameter_um = largest_diameter / 10**3
                individual_pre_post_diameters.append(largest_diameter_um)
                

                # Calculate volume/area based on projections using PCA on presynaptic partner coordinates
                # PCA to get an approximate area of the volume
                pre_mean = np.mean(pre_xyz, axis=0)
                pre_centered_points = pre_xyz - pre_mean
                pre_cov_matrix = np.cov(pre_centered_points, rowvar=False)
                pre_eigenvalues, pre_eigenvectors = np.linalg.eigh(pre_cov_matrix)
                pre_normal_vector = pre_eigenvectors[:, [1, 2]]  # PC2 and PC3

                # Calculate volume/area based on projections using PCA on postsynaptic partner coordinates
                temp_post_coords = curr_post_inputs['pre_pt_position'].tolist()
                post_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_post_coords])
                post_mean = np.mean(post_xyz, axis=0)
                post_centered_points = post_xyz - post_mean
                post_cov_matrix = np.cov(post_centered_points, rowvar=False)
                post_eigenvalues, post_eigenvectors = np.linalg.eigh(post_cov_matrix)
                post_normal_vector = post_eigenvectors[:, [0, 1]]  # PC1 and PC2

                # Project the points
                projected_points = pre_centered_points.dot(post_normal_vector)
                individual_pre_projected_points_ls.append(projected_points)

                # Calculate area
                hull = ConvexHull(projected_points)
                individual_hull_ls.append(hull)
                area = hull.volume  # Area is calculated as volume in 2D
                area_um2 = area / 10**6
                individual_pre_post_areas.append(area_um2)
                
                # Calculate largest diameter
                largest_diameter = 0
                for simplex in hull.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            # Calculate distance between two points
                            d = distance.euclidean(projected_points[simplex[i]], projected_points[simplex[j]])
                            if d > largest_diameter:
                                largest_diameter = d

                # Convert largest diameter to micrometers
                largest_diameter_um = largest_diameter / 10**3
                individual_pre_post_diameters_projected.append(largest_diameter_um)
                
                

            
            

    # Summary data frames
    spatial_span_df = pd.DataFrame()
    spatial_span_df['bodyId_post'] = up_to_date_post_ids
    spatial_span_df['Volume'] = pre_post_volumes
    spatial_span_df['Area'] = pre_post_areas
    spatial_span_df['Diameter'] = pre_post_diameters
    spatial_span_df['Diameter_projected'] = pre_post_diameters_projected
    spatial_span_df['Hull'] = hull_ls
    spatial_span_df['Pre_count'] = pre_count
    spatial_span_df['Pre_xyz'] = pre_xzy_ls
    spatial_span_df['Pre_center'] = pre_center_ls
    spatial_span_df['Post_xyz'] = post_xzy_ls
    spatial_span_df['pre_projected_points'] = pre_projected_points_ls
    spatial_span_df.set_index('bodyId_post', inplace=True)
    spatial_span_df['Area_zscore'] = (spatial_span_df['Area'] - spatial_span_df['Area'].mean()) / spatial_span_df['Area'].std()
    spatial_span_df['Num_pre_sites'] = num_pre_sites
    spatial_span_df['Num_columns'] = [round(area / single_column_area) if area is not None else None for area in pre_post_areas]
    spatial_span_df['Column_span'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in pre_post_diameters]
    spatial_span_df['Column_span_projected'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in pre_post_diameters_projected]
    
    individual_spatial_span_df = pd.DataFrame()
    individual_spatial_span_df['bodyId_post'] = individual_curr_post
    individual_spatial_span_df['Volume'] = individual_pre_post_volumes
    individual_spatial_span_df['Area'] = individual_pre_post_areas
    individual_spatial_span_df['Diameter'] = individual_pre_post_diameters
    individual_spatial_span_df['Diameter_projected'] = individual_pre_post_diameters_projected
    individual_spatial_span_df['Hull'] = individual_hull_ls
    individual_spatial_span_df['Pre_count'] = individual_pre_count
    individual_spatial_span_df['Pre_xyz'] = individual_pre_xzy_ls
    individual_spatial_span_df['Pre_center'] = individual_pre_center_ls
    individual_spatial_span_df['Post_xyz'] = individual_post_xzy_ls
    individual_spatial_span_df['pre_projected_points'] = individual_pre_projected_points_ls
    individual_spatial_span_df.set_index('bodyId_post', inplace=True)
    individual_spatial_span_df['Area_zscore'] = (individual_spatial_span_df['Area'] - individual_spatial_span_df['Area'].mean()) / individual_spatial_span_df['Area'].std()
    individual_spatial_span_df['Num_pre_sites'] = individual_num_pre_sites
    individual_spatial_span_df['Num_columns'] = [round(area / single_column_area) if area is not None else None for area in individual_pre_post_areas]
    individual_spatial_span_df['Column_span'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in individual_pre_post_diameters]
    individual_spatial_span_df['Column_span_projected'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in individual_pre_post_diameters_projected]

    return spatial_span_df, individual_spatial_span_df