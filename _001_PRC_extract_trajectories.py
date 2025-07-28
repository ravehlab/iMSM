import RMF
import numpy as np
from functools import partial
import multiprocessing as mp
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


###########################
#          KAPS           #
###########################

def load_kap_data(input_rmf_path, kap_radius, kap_amount, start_t, end_t, step_t, frames_per_file=104, one_frame_from_each=False):
    """
    Returns: np.array(kap_amount, 3, n_frames)
    """
    trajectories_shape = [kap_amount, 3, int(((end_t - start_t) / step_t) * frames_per_file)]
    trajectories = np.zeros(shape=trajectories_shape)

    for rmf_t in range(start_t, end_t, step_t):
        in_fh = RMF.open_rmf_file_read_only(f"{input_rmf_path}/{rmf_t}.movie.rmf")
        rff = RMF.ReferenceFrameFactory(in_fh)
        tf = RMF.TypedFactory(in_fh)
        kap_string = f"kap{kap_radius}"
        kap_types = [kap_string]

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
                coord = rff.get(type2chains[kap_string][0][kap_i]).get_translation()
                
                trajectories[kap_i, 0, traj_i] = coord[0] / 10
                trajectories[kap_i, 1, traj_i] = coord[1] / 10
                trajectories[kap_i, 2, traj_i] = coord[2] / 10
            if one_frame_from_each:
                break
    return trajectories


def _load_kap_data_worker(i, input_rmf_path, kap_radius, kap_amount, start_t, end_t, step_t, frames_per_file, one_frame_from_each):
    trajectories = load_kap_data(
        input_rmf_path=f"{input_rmf_path}/{i}",
        kap_radius=kap_radius,
        kap_amount=kap_amount,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each)
    return trajectories

def multi_load_kap_data(input_rmf_path, kap_radius, kap_amount, start_t, end_t, step_t, sims_range, frames_per_file=104, one_frame_from_each=False):

            
    load_kap_data_partial = partial(
        _load_kap_data_worker,
        input_rmf_path=input_rmf_path,
        kap_radius=kap_radius,
        kap_amount=kap_amount,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each
    )
    
    good_sims = []
    for i in sims_range:
        # check that trajectory reached end_t time
        if os.path.isfile(f"{input_rmf_path}/{i}/{end_t}.movie.rmf"):        
            good_sims.append(i)
        else:
            print(f"sim {i} not long enough")
    
    num_cores = len(os.sched_getaffinity(0))
    print(f"Using {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        # Map the processing function to all good simulations
        results = pool.map(load_kap_data_partial, good_sims)
        
    return np.concatenate(results, axis=0)


###########################
#          FGS            #
###########################

FG_TYPES = ['Nup2', 'Nsp1', 'Nup100', 'Nup116', 'Nup159', 'Nup49', 'Nup57', 'Nup145', 'Nup1', 'Nup60']
N_CHAINS_PER_FG = [16, 48, 16, 16, 16, 32, 32, 16, 8, 16]
N_BEADS_PER_FG = [20, 32, 40, 48, 34, 14, 15, 13, 44, 12]

def load_fg_data(input_rmf_path, fg_types, n_chains_per_fg, n_beads_per_fg, start_t, end_t, step_t, frames_per_file=104, one_frame_from_each=False):
    """
    Returns: Dictionary(nup_type : np.array([chains, beads, 3, n_frames]))
    """
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
                        coord = rff.get(type2chains[fg_type][chain_i][bead_i]).get_translation()
                        trajectories[fg_type][chain_i, bead_i, 0, traj_i] = coord[0] / 10
                        trajectories[fg_type][chain_i, bead_i, 1, traj_i] = coord[1] / 10
                        trajectories[fg_type][chain_i, bead_i, 2, traj_i] = coord[2] / 10
            if one_frame_from_each:
                break
    return trajectories


def _load_fg_data_worker(i, input_rmf_path, fg_types, n_chains_per_fg, n_beads_per_fg, start_t, end_t, step_t, frames_per_file, one_frame_from_each):
    trajectories = load_fg_data(
        input_rmf_path=f"{input_rmf_path}/{i}",
        fg_types=fg_types,
        n_chains_per_fg=n_chains_per_fg,
        n_beads_per_fg=n_beads_per_fg,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each)
    return trajectories

def multi_load_fg_data(input_rmf_path, fg_types, n_chains_per_fg, n_beads_per_fg, start_t, end_t, step_t, sims_range, frames_per_file=104, one_frame_from_each=False):
    load_fg_data_partial = partial(
        _load_fg_data_worker,
        input_rmf_path=input_rmf_path,
        fg_types=fg_types,
        n_chains_per_fg=n_chains_per_fg,
        n_beads_per_fg=n_beads_per_fg,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each
    )
    
    good_sims = []
    for i in sims_range:
        # check that trajectory reached end_t time
        if os.path.isfile(f"{input_rmf_path}/{i}/{end_t}.movie.rmf"):        
            good_sims.append(i)
        else:
            print(f"sim {i} not long enough")
    
    num_cores = len(os.sched_getaffinity(0))
    print(f"Using {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        # Map the processing function to all good simulations
        results = pool.map(load_fg_data_partial, good_sims)
        
    # Concatenate results
    # trajectories = {fg_type : [] for fg_type in fg_types}
    # for result in results:
    #     for key, val in result.items():
    #         trajectories[key].append(val)
    # for key, val in trajectories.items():
    #     trajectories[key] = np.concatenate(val, axis=0)

    return results