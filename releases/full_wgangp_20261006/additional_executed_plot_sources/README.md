# HUP060 current original-layout renderer

`plot_control_original_style.py` is the exact executed source, SHA-256
`00dccde1fd62fb8e633f296cd9c1373ec7107ca13ff0b1aa01b454d120cda002`.
It was recovered from the executed C-drive figure workflow and is a new
synchronization supplement, not a file claimed to have been present in the
already sealed 943-file adopted archive.

`plot_control_original_style_portable.py` is a path-only adaptation. Its sole
function-body change, immediately after importing `--run-script`, is:

```python
if hasattr(module, "load_runner"):
    module = module.load_runner()
```

This allows the already audited private `reproduce_portable.py` loader to bind
the exact plant, reference, canonical source and setup to that bundle. No plot
math, density estimate, axis, colour, layout, channel, W1, checkpoint or model
equation changes. It has been syntax/diff checked, not used to train or redraw
the adopted figures during repository synchronization. The executed original
remains separate and unmodified.

## Explicit redraw route, using separately supplied private arrays

For the current author's complete adopted bundle, this PowerShell command
illustrates all arguments. Choose a new output directory and run it only when
you intend to redraw. Repository synchronization does not run this command.

```powershell
$bundle = 'D:/论文修改资料/2026-10-06/Full_WGANGP_all_cases_adopted_20261006'
python -B releases/full_wgangp_20261006/additional_executed_plot_sources/plot_control_original_style_portable.py `
  --input "$bundle/supporting_materials/HUP060_restart/runs/fresh_seed20261011_e1000/evaluation/paired_comparison.npz" `
  --paper "$bundle/supporting_materials/HUP060_restart/repro_inputs/paper_baseline_paired_comparison.npz" `
  --run-script "$bundle/supporting_materials/HUP060_restart/reproduce_portable.py" `
  --representative-source "$bundle/supporting_materials/HUP060/output/part3/source_data/figures_06_08/source_data_representative_densities.csv" `
  --all-source "$bundle/supporting_materials/HUP060/current_figure_renderers/source_inputs/local_archive/github_release_taming_epilepsy/output/part3/source_data/figure_08/source_data_actor_wgan_all36_densities.csv" `
  --output './new_HUP060_redraw_output'
```

The representative grid source has columns
`node,role,standardized_amplitude,series,density`, SHA
`ef3499e69b5bca8cab0534d9e5536496a2dff157d9190d4b23431384c6e5ce9a`.
The all-contact grid source has columns
`channel,series,standardized_amplitude,density`, SHA
`63c38d8148b765957f01e9d2fedfd683b998b5064b66f9c7bdd61740813bd8fb`.
The latter `channel` column is required; do not replace it with the newer
`source_data_control_densities.csv` with `node/plot` columns. Both historical
files define only the original plotting grids. Controlled values and W1 come
from the current fresh update-700 paired evaluation, not the historical curves.

The `.npz` inputs are not included in this public supplement. Exact dependency
SHA values remain in the complete private archive and public provenance.
