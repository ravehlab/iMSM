"""
Utilities for extracting KAP and FG trajectories from RMF movie files.

This module provides helper functions to traverse RMF node trees and to load
simulation trajectories for KAPs (karyopherins) and FG-Nups across a range of
RMF movie frames and simulations (optionally in parallel).
"""

from __future__ import annotations

# Standard library imports
import os
import multiprocessing as mp
from functools import partial
from typing import Any, Dict, List, Sequence

# Third-party imports
import numpy as np
import RMF


def _has_depth_with_site(root: Any, i: int) -> bool:
    """Return True if the subtree through the first child has at least depth ``i``
    (including the root node itself) and the leaf is a node named "site".

    Parameters
    ----------
    root : Any
        Root node handle of an RMF tree.
    i : int
        Required depth along the first-child path (1 means current node).

    Returns
    -------
    bool
        True if the first-child chain has depth ``i`` and ends with a "site" node.
    """
    # print(root, i, len(root.get_children()))
    if (i == 1) and root.get_name() == "site":
        return True
    children = root.get_children()
    if len(children) == 0:
        return False
    return _has_depth_with_site(children[0], i - 1)


def _add_nodes(node: Any, tf: Any, type_prefixes: Sequence[str], depth: int = 0) -> List[List[Any]]:
    """Collect node lists whose first child matches any given type prefix.

    Parameters
    ----------
    node : Any
        RMF node to scan.
    tf : Any
        RMF.TypedFactory instance.
    type_prefixes : Sequence[str]
        Full type prefixes to match (e.g., ["Nup1"] for types like "Nup1N").
    depth : int, optional
        Traversal depth used internally for recursion.

    Returns
    -------
    list[list[Any]]
        A list of child-node lists, each corresponding to a matching chain.
    """
    children = node.get_children()
    ret: List[List[Any]] = []
    if len(children) == 0:
        return ret
    if _has_depth_with_site(node, 3) and tf.get_is(children[0]):
        child_type = tf.get(children[0]).get_type_name()
        if any(child_type.startswith(tp) for tp in type_prefixes):
            ret.append(children)
    for c in children:
        ret += _add_nodes(c, tf, type_prefixes, depth + 1)
    return ret

###########################
#          KAPS           #
###########################

def load_kap_data(
    input_rmf_path: str,
    kap_radius: int,
    kap_amount: int,
    start_t: int,
    end_t: int,
    step_t: int,
    frames_per_file: int = 104,
    one_frame_from_each: bool = False,
    get_nsites: int = 0,
) -> np.ndarray:
    """Load KAP coordinates for a single simulation.

    Parameters
    ----------
    input_rmf_path : str
        Path to a single simulation directory containing RMF movie files.
    kap_radius : int
        KAP radius, used to form the KAP type name (e.g., "kap6").
    kap_amount : int
        Number of KAPs expected in the simulation.
    start_t, end_t, step_t : int
        Time indices to iterate over (RMF movie filenames are ``{t}.movie.rmf``).
    frames_per_file : int, optional
        Number of frames within each RMF movie file, by default 104.
    one_frame_from_each : bool, optional
        If True, reads only the first frame from each RMF file, by default False.
    get_nsites : int, optional
        If > 0, also extract up to ``get_nsites`` site coordinates per KAP.

    Returns
    -------
    numpy.ndarray
        If ``get_nsites == 0``: shape (kap_amount, 3, n_frames)
        If ``get_nsites  > 0``: shape (kap_amount, get_nsites, 3, n_frames)
    """
    total_frames = int(((end_t - start_t) / step_t) * frames_per_file)

    if get_nsites != 0:
        trajectories_shape = [kap_amount, get_nsites, 3, total_frames]
    else:
        trajectories_shape = [kap_amount, 3, total_frames]

    trajectories = np.zeros(shape=trajectories_shape)
    for rmf_t in range(start_t, end_t, step_t):
        in_fh = RMF.open_rmf_file_read_only(f"{input_rmf_path}/{rmf_t}.movie.rmf")
        rff = RMF.ReferenceFrameFactory(in_fh)
        bf = RMF.BallFactory(in_fh)
        tf = RMF.TypedFactory(in_fh)
        kap_string = f"kap{kap_radius}"
        kap_types = [kap_string]

        # Load RMF chains indexed by type
        # Note: currently only one KAP type at a time is supported
        type2chains: Dict[str, List[List[Any]]] = {}
        for kap_type in kap_types:
            type2chains[kap_type] = _add_nodes(in_fh.get_root_node(), tf, [kap_type])

        # Iterate frames within the RMF file
        for f_id, f in enumerate(in_fh.get_frames()):
            in_fh.set_current_frame(f)

            traj_i = int(f_id + ((rmf_t - start_t) / step_t) * frames_per_file)

            # Read data for each KAP
            for kap_i in range(kap_amount):
                kap_coord = rff.get(type2chains[kap_string][0][kap_i]).get_translation()

                if get_nsites != 0:
                    # Note: usually site radius is 6A
                    sites = type2chains[kap_string][0][kap_i].get_children()
                    for site_i in range(get_nsites):
                        site_coord = bf.get(sites[site_i]).get_coordinates()
                        trajectories[kap_i, site_i, 0, traj_i] = (site_coord[0] + kap_coord[0]) / 10
                        trajectories[kap_i, site_i, 1, traj_i] = (site_coord[1] + kap_coord[1]) / 10
                        trajectories[kap_i, site_i, 2, traj_i] = (site_coord[2] + kap_coord[2]) / 10
                else:
                    trajectories[kap_i, 0, traj_i] = kap_coord[0] / 10
                    trajectories[kap_i, 1, traj_i] = kap_coord[1] / 10
                    trajectories[kap_i, 2, traj_i] = kap_coord[2] / 10

            if one_frame_from_each:
                break
    return trajectories


def _load_kap_data_worker(
    i: int,
    input_rmf_path: str,
    kap_radius: int,
    kap_amount: int,
    start_t: int,
    end_t: int,
    step_t: int,
    frames_per_file: int,
    one_frame_from_each: bool,
    get_nsites: int,
) -> np.ndarray:
    """Worker to load KAP data for a single simulation index ``i``.

    This exists to support parallel mapping across multiple simulations.
    """
    trajectories = load_kap_data(
        input_rmf_path=f"{input_rmf_path}/{i}",
        kap_radius=kap_radius,
        kap_amount=kap_amount,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each,
        get_nsites=get_nsites,
    )
    return trajectories


def multi_load_kap_data(
    input_rmf_path: str,
    kap_radius: int,
    kap_amount: int,
    start_t: int,
    end_t: int,
    step_t: int,
    sims_range: Sequence[int],
    frames_per_file: int = 104,
    one_frame_from_each: bool = False,
    get_nsites: int = 0,
) -> List[np.ndarray]:
    """Load KAP data across multiple simulations (in parallel).

    Parameters
    ----------
    input_rmf_path : str
        Path containing multiple simulation subdirectories (e.g., ``.../sim_id``).
    kap_radius : int
        KAP radius, used to form the KAP type name (e.g., "kap6").
    kap_amount : int
        Number of KAPs expected in each simulation.
    start_t, end_t, step_t : int
        Time indices to iterate over (RMF movie filenames are ``{t}.movie.rmf``).
    sims_range : Sequence[int]
        Simulation indices to attempt loading.
    frames_per_file : int, optional
        Number of frames within each RMF movie file, by default 104.
    one_frame_from_each : bool, optional
        If True, reads only the first frame from each RMF file, by default False.
    get_nsites : int, optional
        If > 0, also extract up to ``get_nsites`` site coordinates per KAP.

    Returns
    -------
    list[numpy.ndarray]
        A list of trajectory arrays (one per successfully loaded simulation).
    """
    load_kap_data_partial = partial(
        _load_kap_data_worker,
        input_rmf_path=input_rmf_path,
        kap_radius=kap_radius,
        kap_amount=kap_amount,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each,
        get_nsites=get_nsites,
    )

    # Filter simulations that reached end_t
    good_sims: List[int] = []
    for i in sims_range:
        if os.path.isfile(f"{input_rmf_path}/{i}/{end_t}.movie.rmf"):
            good_sims.append(i)
        else:
            print(f"sim {i} not long enough")

    num_cores = len(os.sched_getaffinity(0))
    print(f"Using {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(load_kap_data_partial, good_sims)

    # Note: concatenation across simulations can be performed by the caller if needed.
    return results


# DEBUG VERSION - RUNS WITHOUT MULTIPROCESSING
# def multi_load_kap_data(input_rmf_path, kap_radius, kap_amount, start_t, end_t, step_t, sims_range, frames_per_file=104, one_frame_from_each=False):
#     from functools import partial
#     import os
#
#     load_kap_data_partial = partial(
#         _load_kap_data_worker,
#         input_rmf_path=input_rmf_path,
#         kap_radius=kap_radius,
#         kap_amount=kap_amount,
#         start_t=start_t,
#         end_t=end_t,
#         step_t=step_t,
#         frames_per_file=frames_per_file,
#         one_frame_from_each=one_frame_from_each
#     )
#
#     good_sims = []
#     for i in sims_range:
#         if os.path.isfile(f"{input_rmf_path}/{i}/{end_t}.movie.rmf"):
#             good_sims.append(i)
#         else:
#             print(f"sim {i} not long enough")
#
#     print("Running without multiprocessing (debug mode)")
#     results = []
#     for i in good_sims:
#         try:
#             res = load_kap_data_partial(i)
#         except Exception as e:
#             print(f"Error processing sim {i}: {e}")
#             raise
#         results.append(res)
#
#     return results

###########################
#          FGS            #
###########################

# Default FG-Nup configuration for convenience
FG_TYPES: List[str] = [
    "Nup2",
    "Nsp1",
    "Nup100",
    "Nup116",
    "Nup159",
    "Nup49",
    "Nup57",
    "Nup145",
    "Nup1",
    "Nup60",
]
N_CHAINS_PER_FG: List[int] = [16, 48, 16, 16, 16, 32, 32, 16, 8, 16]
N_BEADS_PER_FG: List[int] = [20, 32, 40, 48, 34, 14, 15, 13, 44, 12]


def load_fg_data(
    input_rmf_path: str,
    fg_types: Sequence[str],
    n_chains_per_fg: Sequence[int],
    n_beads_per_fg: Sequence[int],
    start_t: int,
    end_t: int,
    step_t: int,
    frames_per_file: int = 104,
    one_frame_from_each: bool = False,
) -> Dict[str, np.ndarray]:
    """Load FG-Nup bead coordinates for a single simulation.

    Parameters
    ----------
    input_rmf_path : str
        Path to a single simulation directory containing RMF movie files.
    fg_types : Sequence[str]
        FG-Nup type names.
    n_chains_per_fg : Sequence[int]
        Number of chains per FG type, aligned with ``fg_types``.
    n_beads_per_fg : Sequence[int]
        Number of beads per chain for each FG type, aligned with ``fg_types``.
    start_t, end_t, step_t : int
        Time indices to iterate over (RMF movie filenames are ``{t}.movie.rmf``).
    frames_per_file : int, optional
        Number of frames within each RMF movie file, by default 104.
    one_frame_from_each : bool, optional
        If True, reads only the first frame from each RMF file, by default False.

    Returns
    -------
    dict[str, numpy.ndarray]
        A mapping from FG type to trajectories with shape
        (chains, beads, 3, n_frames).
    """
    trajectories: Dict[str, np.ndarray] = {}
    total_frames = int(((end_t - start_t) / step_t) * frames_per_file)

    for i, fg_type in enumerate(fg_types):
        shape = [n_chains_per_fg[i], n_beads_per_fg[i], 3, total_frames]
        trajectories[fg_type] = np.zeros(shape=shape)

    for rmf_t in range(start_t, end_t, step_t):
        in_fh = RMF.open_rmf_file_read_only(f"{input_rmf_path}/{rmf_t}.movie.rmf")
        rff = RMF.ReferenceFrameFactory(in_fh)
        tf = RMF.TypedFactory(in_fh)

        # Map each FG type to its chain nodes
        type2chains: Dict[str, List[List[Any]]] = {}
        for fg in fg_types:
            type2chains[fg] = _add_nodes(
                in_fh.get_root_node(), tf, [fg + "anchor", fg + "s", fg + "C", fg + "N", fg + "m"]
            )

        # Iterate frames within the RMF file
        for f_id, f in enumerate(in_fh.get_frames()):
            # Set current frame
            in_fh.set_current_frame(f)

            traj_i = int(f_id + ((rmf_t - start_t) / step_t) * frames_per_file)

            # Read data
            
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


def _load_fg_data_worker(
    i: int,
    input_rmf_path: str,
    fg_types: Sequence[str],
    n_chains_per_fg: Sequence[int],
    n_beads_per_fg: Sequence[int],
    start_t: int,
    end_t: int,
    step_t: int,
    frames_per_file: int,
    one_frame_from_each: bool,
) -> Dict[str, np.ndarray]:
    """Worker to load FG data for a single simulation index ``i``."""
    trajectories = load_fg_data(
        input_rmf_path=f"{input_rmf_path}/{i}",
        fg_types=fg_types,
        n_chains_per_fg=n_chains_per_fg,
        n_beads_per_fg=n_beads_per_fg,
        start_t=start_t,
        end_t=end_t,
        step_t=step_t,
        frames_per_file=frames_per_file,
        one_frame_from_each=one_frame_from_each,
    )
    return trajectories


def multi_load_fg_data(
    input_rmf_path: str,
    fg_types: Sequence[str],
    n_chains_per_fg: Sequence[int],
    n_beads_per_fg: Sequence[int],
    start_t: int,
    end_t: int,
    step_t: int,
    sims_range: Sequence[int],
    frames_per_file: int = 104,
    one_frame_from_each: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Load FG data across multiple simulations (in parallel).

    Parameters
    ----------
    input_rmf_path : str
        Path containing multiple simulation subdirectories (e.g., ``.../sim_id``).
    fg_types : Sequence[str]
        FG-Nup type names.
    n_chains_per_fg : Sequence[int]
        Number of chains per FG type, aligned with ``fg_types``.
    n_beads_per_fg : Sequence[int]
        Number of beads per chain for each FG type, aligned with ``fg_types``.
    start_t, end_t, step_t : int
        Time indices to iterate over (RMF movie filenames are ``{t}.movie.rmf``).
    sims_range : Sequence[int]
        Simulation indices to attempt loading.
    frames_per_file : int, optional
        Number of frames within each RMF movie file, by default 104.
    one_frame_from_each : bool, optional
        If True, reads only the first frame from each RMF file, by default False.

    Returns
    -------
    list[dict[str, numpy.ndarray]]
        A list of mappings, one per successfully loaded simulation.
    """
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
        one_frame_from_each=one_frame_from_each,
    )

    # Filter simulations that reached end_t
    good_sims: List[int] = []
    for i in sims_range:
        if os.path.isfile(f"{input_rmf_path}/{i}/{end_t}.movie.rmf"):
            good_sims.append(i)
        else:
            print(f"sim {i} not long enough")

    num_cores = len(os.sched_getaffinity(0))
    print(f"Using {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(load_fg_data_partial, good_sims)

    # Caller can concatenate across simulations if needed.
    return results