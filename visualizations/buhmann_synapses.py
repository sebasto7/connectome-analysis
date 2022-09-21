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
from secrets import token_hex
import json
#from FANC_auto_recon.synapses.synaptic_links import to_ng_annotations, downscale

client = CAVEclient('flywire_fafb_production')

#Defaul values
top_left = np.array([65995, 84864, 5400])*[4,4,40]  # [65995, 84864, 5400], [158278, 71889, 2189]
bottom_right = np.array([69399, 86757, 5478])*[4,4,40]  # [69399, 86757, 5478], [171350, 73983, 2237]


def to_ng_annotations(links, input_order='xyz', input_units=(1, 1, 1),
                      voxel_mip_center=None):
    """
    Create a json representation of a set of synaptic links, appropriate for
    pasting into a neuroglancer annotation layer.
    links: Nx6 numpy array representing N pre-post point pairs.
    input_order: 'xyz' (default) or 'zyx'
        Indicate which column order the input array has.
    input_units: (1, 1, 1) (default) or some other 3-tuple
        If your links are in nm, indicate the voxel size in nm. e.g. (4, 4,
        40) or (40, 4, 4) depending on the input order. If your links are
        already in units of voxels, leave this at the default value.
    voxel_mip_center: None or int
        In neuroglancer, an annotation with an integer coordinate value appears
        at the top-left corner of a voxel, not at the center of that voxel.
        Point annotations often make more sense being placed in the middle of
        the voxel. If False, nothing is added and the neuroglancer default of
        integer values pointing to voxel corners is kept. If voxel_mip_center
        is set to 0, 0.5 will be added to each coordinate so that
        integer-valued inputs end up pointing to the middle of the mip0 voxel.
        If set to 1, 1 will be added to point to the middle of the mip1 voxel.
        If set to x, 0.5 * 2^x will be added to point to the middle of the mipx
        voxel.
        The z coordinate is not changed no matter what, since mips only
        downsample x and y.
    """
    assert input_order in ['xyz', 'zyx']

    def line_anno(pre, post):
        return {
            'pointA': [x for x in pre],
            #'pointA': [int(x) for x in pre],
            'pointB': [x for x in post],
            #'pointB': [int(x) for x in post],
            'type': 'line',
            'id': token_hex(40)
        }

    if isinstance(links, str):
        links = load(links)

    if input_units is not (1, 1, 1):
        links = downscale(links.astype(float), input_units, inplace=False)
        # Now links are in units of voxels

    if input_order == 'zyx':
        links = flip_xyz_zyx_convention(links, inplace=False)

    if voxel_mip_center is not None:
        delta = 0.5 * 2**voxel_mip_center
        adjustment = (delta, delta, 0, delta, delta, 0)
        links = links.astype(float) + adjustment

    annotations = [line_anno(links[i, 0:3],
                             links[i, 3:6])
                   for i in range(links.shape[0])]
    #print(json.dumps(annotations, indent=2))

    try:
        import pyperclip
        answer = input("Want to copy the output to the clipboard? (Only works if "
                       "you're running this script on a local machine, not on a "
                       "server.) [y/n] ")
        if answer.lower() == 'y':
            print('Copying')
            pyperclip.copy(json.dumps(annotations))
    except:
        print("Install pyperclip (pip install pyperclip) for the option to"
              " programmatically copy the output above to the clipboard")

    return json.dumps(annotations, indent=2)


def downscale(array, scale_factor, inplace=True):
    """
    Given an Nx6 array and a scaling factor (constant or 3-length), divide the
    first 3 columns and the last 3 columns of the array by the scale factor.
    """
    if not inplace:
        array = np.copy(array)
    array[:, 0:3] = array[:, 0:3] / scale_factor
    array[:, 3:6] = array[:, 3:6] / scale_factor
    if not inplace:
        return array



if __name__ == "__main__":
    print("Running buhmann_synapses.py")
    pre_id = input('Enter the presynaptic ID:')
    post_id = input('Enter the postsynaptic ID:')
    if pre_id == "" or post_id == "":
        print("Using default example neurons")
        pre_id = 720575940659388801 # L3
        post_id = 720575940626482442 # Tm9

    synapses = client.materialize.synapse_query(
        pre_ids = int(pre_id), post_ids = int(post_id),
        bounding_box_column='pre_pt_position'
        )

    url = make_synapse_neuroglancer_link(
        synapses, client,
        point_column='pre_pt_position',
        link_pre_and_post=True, return_as="url"
        )
    print(f"Presinaptic terminals here: {url}")

    synapses_vectors = np.hstack(
        [np.vstack(synapses.pre_pt_position.values),
        np.vstack(synapses.post_pt_position.values)])

    json_dict = to_ng_annotations(synapses_vectors, input_order='xyz', voxel_mip_center=1, input_units=(4,4,40))
