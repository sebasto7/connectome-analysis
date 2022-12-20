""" This script loads and generate visualizations for Buhmann synapses at desired
regions in the data set

# Requisites:
1. nglui package installed in your environment (if not, go to https://pypi.org/project/nglui/)
2. CAVEclient TOKEN saved in your PC. If not, run:
    # Getting the TOKEN,
    client = caveclient.CAVEclient()
    auth = client.auth
    print(f"My current token is: {auth.token}")
    auth.get_new_token()
    #Saving TOKEN in my PC,
    client.auth.save_token(token="your_token")
"""

import numpy as np
from caveclient import CAVEclient
from nglui.statebuilder.helpers import make_synapse_neuroglancer_link
from FANC_synaptic_links import to_ng_annotations, downscale
from helpers.synapse_queries import combine_xyz
from fafbseg import flywire

client = CAVEclient('flywire_fafb_production')

#Defaul values
top_left = np.array([65995, 84864, 5400])*[4,4,40]  # [65995, 84864, 5400], [158278, 71889, 2189]
bottom_right = np.array([69399, 86757, 5478])*[4,4,40]  # [69399, 86757, 5478], [171350, 73983, 2237]
defaul_pre_id = 720575940659388801 # L3
defaul_post_id = 720575940626482442 # Tm9


if __name__ == "__main__":
    print("Running buhmann_synapses.py")
    pre_id = input('Enter the presynaptic ID:')
    post_id = input('Enter the postsynaptic ID:')
    method = input("Enter method to use, 'FAFB' or 'CAVE':")
    if pre_id == "" or post_id == "":
        print("Using default example neurons")
        pre_id = defaul_pre_id # L3
        post_id = defaul_post_id# Tm9

    #Updating segment IDs
    pre_id = client.chunkedgraph.get_latest_roots(pre_id).tolist()
    post_id = client.chunkedgraph.get_latest_roots(post_id).tolist()
    new_segmentsIDs_ls = [x for y in zip(pre_id, post_id) for x in y] #Combining lists of equal size
    print(f'Update ids, pre: {pre_id}, post: {post_id}')

    if method == "FAFB":

        synapses = flywire.synapses.fetch_synapses(post_id,
                        pre=False, post=True, attach=True,
                        min_score=50, clean=True, transmitters=False,
                        neuropils=True, batch_size=30,
                        dataset='production', progress=True,mat="live")

        # TODO: implement code in case the "pre_id"  is a list of more than one item.
        # This happens when a segment ID is split in two or more.
        # Check for each of the new segemnt IDs, which is the one in contact with the "post_id"

        synapses = synapses[synapses['pre'] == pre_id[0]]
        #Combining x,y,z columns for future purposes
        combine_xyz(synapses)
        # Getting URL
        url = make_synapse_neuroglancer_link(synapses, client,
                point_column='post_pt_position',return_as="url")

    elif method == "CAVE":

        synapses = client.materialize.synapse_query(
            pre_ids = int(pre_id[0]), post_ids = int(post_id[0]),
            bounding_box_column='pre_pt_position')

        url = make_synapse_neuroglancer_link(
            synapses, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url")

    print(f"Site are here as annotations: {url}")

    # # Linking pre and post sites:
    # synapses_vectors = np.hstack(
    #     [np.vstack(synapses.pre_pt_position.values),
    #     np.vstack(synapses.post_pt_position.values)])
    #
    # json_dict = to_ng_annotations(synapses_vectors, input_order='xyz', voxel_mip_center=1, input_units=(4,4,40))
