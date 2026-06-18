# %% imports
import os
import importlib   

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot

import figure_stationary_helpers
importlib.reload(figure_stationary_helpers)
from figure_stationary_helpers import visualize_stationary_distribution

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

