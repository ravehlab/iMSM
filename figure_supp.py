# %% imports
import os
import importlib   

import figure_supp_helpers
importlib.reload(figure_supp_helpers)
from figure_supp_helpers import plot_states_found_over_subset, plot_implied_timescales, plot_committor_vs_z

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot, comparison_vertical_plot

import figure_permeability_helpers
importlib.reload(figure_permeability_helpers)
from figure_permeability_helpers import plot_speedup_factor_heatmap, plot_all_convergences, plot_permeability_pore_diameter_comparison

import figure_free_energy_helpers
importlib.reload(figure_free_energy_helpers)
from figure_free_energy_helpers import plot_all_1d_landscapes

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

# %% committor vs z (not interesting)
fig = plot_committor_vs_z(
    z_nuc=-15.0,
    z_cyt=15.0
)
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/committor_vs_z"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300, facecolor='none')

# %% convergence sim subset
fig1 = plot_speedup_factor_heatmap(use_sim_subset=True)
save_path1 = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/speedup_factor_sim_subset"
os.makedirs(os.path.dirname(save_path1), exist_ok=True)
fig1.savefig(save_path1 + ".png", bbox_inches='tight', dpi=300)

fig2 = plot_all_convergences(use_sim_subset=True)
save_path2 = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/all_convergences_sim_subset"
os.makedirs(os.path.dirname(save_path2), exist_ok=True)
fig2.savefig(save_path2 + ".png", bbox_inches='tight', dpi=300)

# %% 1d free energy landscapes
fig = plot_all_1d_landscapes()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/1d_landscapes"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% pore diameter permeability comparison
fig = plot_permeability_pore_diameter_comparison()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/pore_diameter_permeability_comparison"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)


# %% diameter single spoke colored by nups
tm_paths_base = [
    "data/ntr_variants_46R/4_18/6_transition_matrices_subsets/1.00fraction/0index/320clusters.pickle",
    "data/ntr_variants/4_18_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    "data/ntr_variants_62R/4_18/6_transition_matrices_subsets/1.00fraction/0index/320clusters.pickle",
    "data/ntr_variants_70R/4_18/6_transition_matrices_subsets/1.00fraction/0index/320clusters.pickle"
]

cluster_paths_base = [
    "data/ntr_variants_46R/4_18/5_clustering_subsets/1.00fraction/0index/320clusters.pickle",
    "data/ntr_variants/4_18_more/5_clustering_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    "data/ntr_variants_62R/4_18/5_clustering_subsets/1.00fraction/0index/320clusters.pickle",
    "data/ntr_variants_70R/4_18/5_clustering_subsets/1.00fraction/0index/320clusters.pickle"
]

titles_base = ["46 nm Pore", "54 nm Pore", "62 nm Pore", "70 nm Pore"]

fig_b = comparison_vertical_plot(
    tm_paths=tm_paths_base,
    cluster_paths=cluster_paths_base,
    titles=titles_base,
    n_rows=1,
    n_cols=4,
    spoke_index=0,
    time_step_us=5.0,
    min_rate=0.0005,
    connect_threshold=0.05,
    save_path=None,
    title=None,
    pie_scaling=25,
    neighbor_only=False,
    rate_based_thickness=True,
    xlim=(-16, 16),
    color_by="nup",
    show_arrows=True,
    show_stationary_dist=False
)
save_path_b = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_supp/single_spoke_diameter_nup_color"
os.makedirs(os.path.dirname(save_path_b), exist_ok=True)
fig_b.savefig(save_path_b + ".png", bbox_inches='tight', dpi=300)


