# %% imports
import os
import importlib   

import figure_free_energy_helpers
importlib.reload(figure_free_energy_helpers)
from figure_free_energy_helpers import generate_free_energy_data, plot_z0_free_energy_vs_mw

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot_custom, comparison_vertical_plot

# %% panel a
# generate_free_energy_data()
fig = plot_z0_free_energy_vs_mw()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_5/panel_a"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)


# %% panels b c d
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

# Panel b (color by Z, show arrows, hide stationary dist, show pore residency)
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
    color_by="z",
    show_arrows=True,
    show_stationary_dist=False,
    legend_bbox_to_anchor=(0.05, 0.98),
    show_pore_residency=True
)
save_path_b = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_5/panel_b"
os.makedirs(os.path.dirname(save_path_b), exist_ok=True)
fig_b.savefig(save_path_b + ".png", bbox_inches='tight', dpi=300)

# Panel c (color by Type, show arrows with 25% opacity, hide stationary dist)
fig_c = comparison_vertical_plot(
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
    color_by="type",
    show_arrows=True,
    arrow_opacity=0.25,
    show_stationary_dist=False,
    legend_bbox_to_anchor=(0.05, 0.96),
    show_colorbar=False
)
save_path_c = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_5/panel_c"
os.makedirs(os.path.dirname(save_path_c), exist_ok=True)
fig_c.savefig(save_path_c + ".png", bbox_inches='tight', dpi=300)

# Panel d (color by Committor, show arrows with 25% opacity, show stationary dist)
fig_d = comparison_vertical_plot(
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
    color_by="committor",
    show_arrows=True,
    arrow_opacity=0.25,
    show_stationary_dist=True,
    legend_bbox_to_anchor=(0.05, 0.96),
    show_colorbar=False
)
save_path_d = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_5/panel_d"
os.makedirs(os.path.dirname(save_path_d), exist_ok=True)
fig_d.savefig(save_path_d + ".png", bbox_inches='tight', dpi=300)