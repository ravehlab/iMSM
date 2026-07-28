# %% imports
import os
import importlib   

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot, comparison_vertical_plot

import figure_stationary_helpers
importlib.reload(figure_stationary_helpers)
from figure_stationary_helpers import visualize_stationary_distribution, plot_initiator_nups

import figure_permeability_helpers
importlib.reload(figure_permeability_helpers)
from figure_permeability_helpers import plot_4_6_cmp, save_convergence_data

# %% panel a

fig = comparison_plot(
    base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{320}clusters.pickle",
    base_cluster_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{320}clusters.pickle",
    radii=[26],
    n_sites=[4,6],
    title='Comparison of transport graphs for 10us window size, 160 clusters'
)
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_a"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel b

n_clusters = 320
r = 10
for n_sites in [4, 6]:
    fig = visualize_stationary_distribution(
        base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
        base_clustering_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
        base_clustered_path=f"data/ntr_variants/#n#_#r#_more/5_clustered_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
        r = r,
        n = n_sites,
        title = f"Stationary distribution for r={r}A, n={n_sites} sites, MSM w/ {n_clusters} clusters",
        # x_limits = (0,0.045),
        x_limits = (0, 7),
        use_energy=True,
        add_number_labels = False,
        add_bead_amounts=False,
        include_unbound_states=True,
        include_mid_channel=False,
        right_to_left=True if n_sites == 4 else False,
        center_labels=True
    )
    save_path = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_b_{n_sites}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)


# %% panel c
# save_convergence_data()
fig = plot_4_6_cmp()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_c"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel d
fig = plot_initiator_nups(
    base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{320}clusters.pickle",
    base_clustering_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{320}clusters.pickle"
)
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_d"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)


# %% panel e

tm_paths = [
    "data/ntr_variants/4_26_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    "data/ntr_variants/6_26_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/320clusters.pickle"
]

cluster_paths = [
    "data/ntr_variants/4_26_more/5_clustering_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    "data/ntr_variants/6_26_more/5_clustering_subsets/1.00fraction_simulations/0index/320clusters.pickle"
]

titles = ["4 Sites", "6 Sites"]

# Panel e1 (color by Z, show arrows, hide stationary dist, show pore residency)
fig_e1 = comparison_vertical_plot(
    tm_paths=tm_paths,
    cluster_paths=cluster_paths,
    titles=titles,
    n_rows=1,
    n_cols=2,
    spoke_index=0,
    time_step_us=5.0,
    min_rate=0.000005,
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
    legend_bbox_to_anchor=(0.05, 1.04),
    show_pore_residency=True,
    legend_ncol=2
)
save_path_e1 = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_e1"
os.makedirs(os.path.dirname(save_path_e1), exist_ok=True)
fig_e1.savefig(save_path_e1 + ".png", bbox_inches='tight', dpi=300)

# Panel e2 (color by Type, show arrows with 25% opacity, hide stationary dist)
fig_e2 = comparison_vertical_plot(
    tm_paths=tm_paths,
    cluster_paths=cluster_paths,
    titles=titles,
    n_rows=1,
    n_cols=2,
    spoke_index=0,
    time_step_us=5.0,
    min_rate=0.000005,
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
    legend_bbox_to_anchor=(0.05, 1.02),
    show_colorbar=False,
    show_y_axis=False,
    legend_ncol=1
)
save_path_e2 = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_e2"
os.makedirs(os.path.dirname(save_path_e2), exist_ok=True)
fig_e2.savefig(save_path_e2 + ".png", bbox_inches='tight', dpi=300)

# Panel e3 (color by Committor, show arrows with 25% opacity, show stationary dist)
fig_e3 = comparison_vertical_plot(
    tm_paths=tm_paths,
    cluster_paths=cluster_paths,
    titles=titles,
    n_rows=1,
    n_cols=2,
    spoke_index=0,
    time_step_us=5.0,
    min_rate=0.000005,
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
    legend_bbox_to_anchor=(0.05, 1.02),
    show_colorbar=False,
    show_y_axis=False,
    legend_ncol=2
)
save_path_e3 = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_2/panel_e3"
os.makedirs(os.path.dirname(save_path_e3), exist_ok=True)
fig_e3.savefig(save_path_e3 + ".png", bbox_inches='tight', dpi=300)


