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
