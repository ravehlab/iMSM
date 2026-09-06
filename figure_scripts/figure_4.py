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

import figure_permeability_helpers
importlib.reload(figure_permeability_helpers)
from figure_permeability_helpers import plot_speedup_factor_heatmap, plot_all_convergences, plot_md_vs_msm_convergence



# %% panel a
fig = plot_speedup_factor_heatmap()
save_path = "plots/figure_4/panel_a"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel b
fig = plot_all_convergences()
save_path = "plots/figure_4/panel_b"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)

# %% panel c
fig = plot_md_vs_msm_convergence()
save_path = "plots/figure_4/panel_c"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path + ".png", bbox_inches='tight', dpi=300)