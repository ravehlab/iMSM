# import dependencies
import RMF
import IMP.npctransport
import pickle
import numpy as np
import os

def _has_depth_with_site(root, i):
    """ returns true if node subtree thru first child is at least i
        levels, including the root node itself, and the lead is a site """
#  print root, i, len(root.get_children())
    if (i==1) and root.get_name()=="site":
        return True
    c = root.get_children()
    if len(c) == 0:
        return False
    return _has_depth_with_site(c[0], i-1)

def _add_nodes(node, tf, type_prefixes, depth=0):
    '''
    node - rmf node to scan
    tf - typed factory
    type_prefixes - list of full type prefixes (e.g. "Nup1" for "Nup1N")

    adds only nodes whose type name begins with any of the specified type prefixes
    '''
    children = node.get_children()
    ret = []
    if len(children)==0:
        return ret
    if _has_depth_with_site(node, 3) and tf.get_is(children[0]):
        child_type = tf.get(children[0]).get_type_name()
        if any([child_type.startswith(tp) for tp in type_prefixes]):
            ret.append(children)
    for c in children:
        ret += _add_nodes(c, tf,  type_prefixes, depth+1)
    return ret

def _add_nodes_exact(node, tf, types, depth=0):
    '''
    node - rmf node to scan
    tf - typed factory
    types - list of full types (e.g. "Nup1N")

    adds only nodes whose type name begins with any of the specified type prefixes
    '''
    children = node.get_children()
    ret = []
    if len(children)==0:
        return ret
    if _has_depth_with_site(node, 3) and tf.get_is(children[0]):
        child_type = tf.get(children[0]).get_type_name()
        if any([child_type == tp for tp in types]):
            ret.append(children)
    for c in children:
        ret += _add_nodes_exact(c, tf,  types, depth+1)
    return ret

def get_trajectories_shape(n_molecules, start_t, end_t, step_t, frames_per_file):
    return [n_molecules, 3, int(((end_t - start_t) / step_t) * frames_per_file)]

def load_kap_data(input_rmf_path, kap_radius, kap_amount, start_t, end_t, step_t, frames_per_file=104, one_frame_from_each=False):
    trajectories_shape = [kap_amount, 3, int(((end_t - start_t) / step_t) * frames_per_file)]
    trajectories = np.zeros(shape=trajectories_shape)

    for rmf_t in range(start_t, end_t, step_t):
        in_fh = RMF.open_rmf_file_read_only(f"{input_rmf_path}/{rmf_t}.movie.rmf")
        rff = RMF.ReferenceFrameFactory(in_fh)
        tf = RMF.TypedFactory(in_fh)
        kap_types = [f"kap{kap_radius}"]

        # load data
        type2chains={}
        for i, kap_type in enumerate(kap_types):
            type2chains[kap_type] = _add_nodes(in_fh.get_root_node(), tf, [kap_type])
            
        # set frame
        for f_id, f in enumerate(in_fh.get_frames()):
            in_fh.set_current_frame(f)
    
            traj_i = int(f_id + ((rmf_t - start_t) / step_t) * frames_per_file)
            # read data
            for kap_i in range(kap_amount):
                coord = rff.get(type2chains["kap35"][0][kap_i]).get_translation()
                
                trajectories[kap_i, 0, traj_i] = coord[0] / 10
                trajectories[kap_i, 1, traj_i] = coord[1] / 10
                trajectories[kap_i, 2, traj_i] = coord[2] / 10
            if one_frame_from_each:
                break
    return trajectories

def _get_fg_types(input_pb_path):
    fg_types = []
    output = IMP.npctransport.Output()
    FILE = open(input_pb_path,"rb")
    output.ParseFromString(FILE.read())
    a = output.assignment
    for fg in a.fgs:
        fg_types.append(fg.type)
    return fg_types

def load_fg_data(input_rmf_path, fg_types, n_chains_per_fg, n_beads_per_fg, start_t, end_t, step_t, frames_per_file=104, one_frame_from_each=False):
    trajectories = dict()
    for i, fg_type in enumerate(fg_types):
        shape = [n_chains_per_fg[i], n_beads_per_fg[i], 3, int(((end_t - start_t) / step_t) * frames_per_file)]
        trajectories[fg_type] = np.zeros(shape=shape)

    for rmf_t in range(start_t, end_t, step_t):
        in_fh = RMF.open_rmf_file_read_only(f"{input_rmf_path}/{rmf_t}.movie.rmf")
        rff = RMF.ReferenceFrameFactory(in_fh)
        tf = RMF.TypedFactory(in_fh)
        type2chains={}
        for i, fg in enumerate(fg_types):
            type2chains[fg] = _add_nodes(in_fh.get_root_node(), tf, [fg + "anchor", fg + "s", fg + "C", fg + "N", fg + "m"])
            
        for f_id, f in enumerate(in_fh.get_frames()):
            # set frame
            in_fh.set_current_frame(f)
    
            traj_i = int(f_id + ((rmf_t - start_t) / step_t) * frames_per_file)
            # read data
            for nup_i, fg_type in enumerate(fg_types):
                for chain_i in range(n_chains_per_fg[nup_i]):
                    for bead_i in range(n_beads_per_fg[nup_i]):
                        try:
                            coord = rff.get(type2chains[fg_type][chain_i][bead_i]).get_translation()
                            trajectories[fg_type][chain_i, bead_i, 0, traj_i] = coord[0] / 10
                            trajectories[fg_type][chain_i, bead_i, 1, traj_i] = coord[1] / 10
                            trajectories[fg_type][chain_i, bead_i, 2, traj_i] = coord[2] / 10
                        except:
                            pass
            if one_frame_from_each:
                break
    return trajectories    


fg_types = ['Nsp1', 'Nup100', 'Nup116', 'Nup159', 'Nup49', 'Nup57', 'Nup145', 'Nup1', 'Nup60']
n_chains_per_fg = [48, 16, 16, 16, 32, 32, 16, 72, 16]
n_beads_per_fg = [32, 40, 48, 34, 14, 15, 13, 40, 12]
start_t = 150000
end_t = 160000
step_t = 100
frames_per_file = 1
sims_range = range(1, 51)

# bad_sims = []
# for i in sims_range:
#     # check that trajectory reached end_t time
#     if not os.path.isfile(f"/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Dataset/dataset/full_50uM_1ns/divergences/{i}/{end_t}.movie.rmf"):
#         print(f"sim {i} not long enough")
#         bad_sims.append(i)
#         continue

good_sims = [i for i in sims_range] #if i not in bad_sims]


from concurrent.futures import ThreadPoolExecutor

def process_sim(i):
    print(f"loading sim {i}")
    trajectories = load_fg_data(input_rmf_path=f"/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Dataset/dataset/full_50uM_1ns/divergences/{1}/",
            fg_types=fg_types,
            n_chains_per_fg=n_chains_per_fg,
            n_beads_per_fg=n_beads_per_fg,
            start_t=start_t,
            end_t=end_t,
            step_t=100,
            frames_per_file=1,
            one_frame_from_each=True)
    with open(f"data/singles/{i}/150-160-fgs.pickle", "wb") as f:
        pickle.dump(trajectories, f)

with ThreadPoolExecutor(max_workers=None) as executor:
    executor.map(process_sim, good_sims)

