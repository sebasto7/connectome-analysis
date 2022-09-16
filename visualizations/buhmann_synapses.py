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

client = CAVEclient('flywire_fafb_production')

top_left = np.array([158278, 71889, 2189])*[4,4,40]
bottom_right = np.array([171350, 73983, 2237])*[4,4,40]
synapses = client.materialize.synapse_query(
    bounding_box=[top_left, bottom_right],
    bounding_box_column='pre_pt_position')

if __name__ == "__main__":
    print("Running buhmann_synapses.py")
    url = make_synapse_neuroglancer_link(synapses, client, point_column='pre_pt_position',return_as="url")
    print(url)
