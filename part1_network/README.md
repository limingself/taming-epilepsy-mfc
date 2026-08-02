# Part I: PLV network and actuator selection

This directory contains the final model and plotting code for manuscript
Fig. 1.

- `part1_model.py` is the single Part-I analysis entry.  It reads the frozen
  HUP060 multi-band PLV matrices, constructs a connected maximum-spanning-tree
  backbone plus strongest residual edges, computes the weighted centrality
  score, and applies the fixed threshold to select actuators.
- `figure_01_plv_network_selection.py` reads the resulting Source Data and
  creates the independent Fig. 1 bundle.
- `config.yaml` records the final parameters and paths.
- `hup060_clinical_labels.csv` contains frozen SOZ/resection annotations.
  Labels are used only for retrospective coloring after selection; they do not
  enter the centrality score or threshold rule.

Run from the repository root:

```powershell
python part1_network/part1_model.py
python part1_network/figure_01_plv_network_selection.py
```

Source Data are written to `output/part1/source_data/figure_01/`; figure files
are written to `output/part1/figure_01/`.
