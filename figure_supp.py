# %% imports
import os
import importlib   

import figure_supp_helpers
importlib.reload(figure_supp_helpers)
from figure_supp_helpers import plot_states_found_over_subset, plot_implied_timescales

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot

# %% states_over_subset
fig = plot_states_found_over_subset()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/states_over_subset"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% implied_timesales
fig = plot_implied_timescales()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/implied_timescales"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% in out flows
n_clusters = 320
for flow in ['in', 'out']:
    fig = comparison_plot(
        base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
        base_cluster_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
        radii=[26],
        n_sites=[4,6],
        title=f'{flow} flows of states for r=26A, n=4,6 sites, MSM w/ {n_clusters} clusters',
        in_out_flow=flow
    )
    save_path = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/in_out_flows_{flow}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
