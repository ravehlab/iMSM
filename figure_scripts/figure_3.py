# %% imports
import os
import sys
from pathlib import Path

_fig_dir: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_repo_root: Path = _fig_dir.parent if _fig_dir.name == "figure_scripts" else _fig_dir
for _p in [str(_fig_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists("data") and (_repo_root / "data").exists():
    os.chdir(_repo_root)

import importlib

import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot

import figure_permeability_helpers
importlib.reload(figure_permeability_helpers)
from figure_permeability_helpers import plot_full_permeability_comparison, plot_full_permeability_comparison_alt

import figure_stationary_helpers
importlib.reload(figure_stationary_helpers)
from figure_stationary_helpers import plot_free_energy_grid, plot_js_wd


# %% panel a

fig = comparison_plot(
    base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{320}clusters.pickle",
    base_cluster_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{320}clusters.pickle",
    radii=[10,18,26],
    n_sites=[2,4,6],
    title='Comparison of transport graphs for 5us window size, 160 clusters, 56D',
    min_rate=0.01,
    max_rate=10,
    time_step_us=5,
    in_out_flow=None,
    show_scale_bars=False,
    swap_axes=True,
    add_mini_titles=False,
    add_nucleus_cytoplasm_text=True,
    nucleus_cytoplasm_top_row_only=True,
    nucleus_cytoplasm_fontsize=54
)
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_3/panel_a"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel b
fig = plot_full_permeability_comparison()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_3/panel_b"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

fig = plot_full_permeability_comparison_alt()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_3/panel_b_alt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel c

n_clusters = 320
paths = {
    'tm': f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
    'clustering': f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle",
    'clustered': f"data/ntr_variants/#n#_#r#_more/5_clustered_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle"
}
# Run the function
fig, _ = plot_free_energy_grid(
    n_sites_list=[2, 4, 6],
    r_list=[10, 14, 18, 22, 26],
    paths=paths,
    include_mid_channel=False,
    title='Free Energy Distributions Across Nup Layers, 56 nm diameter'
)

save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_3/panel_c"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel d
fig_js, fig_wd = plot_js_wd()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_3/panel_d"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig_js.savefig(save_path + "_js.png", bbox_inches='tight', dpi=300)
fig_wd.savefig(save_path + "_wd.png", bbox_inches='tight', dpi=300)