# %% imports
import os
import importlib   

import figure_permeability_helpers
importlib.reload(figure_permeability_helpers)
from figure_permeability_helpers import plot_speedup_factor_heatmap, plot_all_convergences, plot_md_vs_msm_convergence



# %% panel a
fig = plot_speedup_factor_heatmap()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_4/panel_a"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel b
fig = plot_all_convergences()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_4/panel_b"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel c
fig = plot_md_vs_msm_convergence()
save_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_4/panel_c"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)