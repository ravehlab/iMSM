# %% imports
import os
import importlib

import figure_fg_sliding_helpers
importlib.reload(figure_fg_sliding_helpers)
from figure_fg_sliding_helpers import (
    setup_fg_sliding_data,
    setup_fg_sliding_params,
    run_fg_sliding_imsm,
    generate_fg_vmd_scripts,
    plot_fg_spatial_network,
)


# %% setup data, parameters, and run iMSM
# data = setup_fg_sliding_data(
#     traj_paths=[
#         "data/nup_sims/fsfgx2/output_from_0_to_2499_nowat_unwarped.dcd",
#         "data/nup_sims/fsfgx2/output_from_2500_to_4999_nowat_unwarped.dcd",
#         "data/nup_sims/fsfgx2/output_from_5000_to_7499_nowat_unwarped.dcd",
#         "data/nup_sims/fsfgx2/output_from_7500_to_9999_nowat_unwarped.dcd",
#         "data/nup_sims/fsfgx2/output_from_10000_to_12499_nowat_unwarped.dcd",
#         "data/nup_sims/fsfgx2/output_from_12500_nowat_unwarped.dcd",
#     ],
#     top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
#     imsm_checkpoint_base_path="data/nup_sims/fsfgx2/imsm/",
#     stride=1,
#     first_frame=None,
#     last_frame=None,
# )

# params = setup_fg_sliding_params(
#     window_size=500,
#     k_closest=5,
#     max_surface_dist=1.0,
#     n_clusters=5,
#     merge_cluster_threshold=0.1,
#     tm_prior=0.01,
#     start_stage=1,
#     end_stage=5,
#     ns_per_frame_base=0.96,
#     stride=1,
# )

# run_fg_sliding_imsm(
#     imsm_runs=data["imsm_runs"],  # type: ignore[arg-type]
#     params=params,
#     top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
#     kap_n_ca=861,
#     heat5_range=(175, 195),
#     heat6_range=(218, 232),
# )


# %% panels a & c

exemplars = generate_fg_vmd_scripts(
    top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    traj_paths=[
        "data/nup_sims/fsfgx2/output_from_0_to_2499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_2500_to_4999_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_5000_to_7499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_7500_to_9999_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_10000_to_12499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_12500_nowat_unwarped.dcd",
    ],
    imsm_checkpoint_path="data/nup_sims/fsfgx2/imsm/fgC88_91_com",
    output_dir="vmd_states/fgC88_91_com",
    kap_n_ca=861,
    focal_fg_alphacarbon=(88, 91),
    window_size=500,
    stride=1,
    first_frame=None,
    ns_per_frame=0.96,
    free_threshold=0.90,
    auto_align_plane=True,
    rotation_x_deg=0.0,
    rotation_y_deg=0.0,
    rotation_z_deg=0.0,
    heat5_range=(175, 195),
    heat6_range=(218, 232),
    vmd_zoom_scale=8.0,
    all_states_translate=(-0.3, 0.3, 0.0),
)


# %% panel b

save_path_b = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_6/panel_b"
os.makedirs(os.path.dirname(save_path_b), exist_ok=True)

fig_b = plot_fg_spatial_network(
    top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    imsm_checkpoint_path="data/nup_sims/fsfgx2/imsm/fgC88_91_com",
    output_plot_path=f"{save_path_b}.png",
    top_rate_percentile=10.0,
    kap_n_ca=861,
    repeat_colors=("#78909C", "#CFD8DC"),
    zoom=True,
    state_size_factor=5.0,
    free_threshold=0.9,
    dt_ns=1000.0,
    auto_align_plane=True,
    rotation_x_deg=0.0,
    rotation_y_deg=0.0,
    rotation_z_deg=0.0,
    heat5_range=(175, 195),
    heat6_range=(218, 232),
    focal_fg_alphacarbon=(88, 91),
    native_pdb_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    k_closest=5,
    max_surface_dist=1.0,
    native_similarity_threshold=0.75,
    top_n_print=10,
    title="",
    show_full_kap=False,
    heat_ribbon_width=30,
    heat_ribbon_alpha=0.25,
    state_color_mode="residue_index",
    arrow_color="black",
    show_rate_colorbar=False,
    show_residue_colorbar=True,
    diverging_palette=(
        "#6B200C",
        "#973D21",
        "#DA6C42",
        "#EE956A",
        "#FBC2A9",
        "#D1D397",
        "#BAD6F9",
        "#7DB0EA",
        "#447FDD",
        "#225BB2",
        "#133E7E",
    ),
    residue_color_range=(1, 21),
)


# %% panel d

save_path_d = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/figure_6/panel_d"
os.makedirs(os.path.dirname(save_path_d), exist_ok=True)

fig_d = plot_fg_spatial_network(
    top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    imsm_checkpoint_path="data/nup_sims/fsfgx2/imsm/fgC88_91_com",
    output_plot_path=f"{save_path_d}.png",
    top_rate_percentile=10.0,
    kap_n_ca=861,
    repeat_colors=("#78909C", "#CFD8DC"),
    zoom=True,
    state_size_factor=5.0,
    free_threshold=0.9,
    dt_ns=1000.0,
    auto_align_plane=True,
    rotation_x_deg=0.0,
    rotation_y_deg=0.0,
    rotation_z_deg=0.0,
    heat5_range=(175, 195),
    heat6_range=(218, 232),
    focal_fg_alphacarbon=(88, 91),
    native_pdb_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    k_closest=5,
    max_surface_dist=1.0,
    native_similarity_threshold=0.75,
    top_n_print=10,
    title="",
    show_full_kap=False,
    heat_ribbon_width=30,
    heat_ribbon_alpha=0.25,
    state_color_mode="interaction_fraction",
    arrow_color=None,
    show_rate_colorbar=True,
    show_residue_colorbar=False,
    diverging_palette=(),
    residue_color_range=(1, 21),
)
