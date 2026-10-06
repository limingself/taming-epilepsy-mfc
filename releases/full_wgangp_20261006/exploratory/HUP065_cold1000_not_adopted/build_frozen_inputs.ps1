$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\LiMing\Documents\改论文\.full_wgangp_all_cases_adopt_20261006\hup065_improvement'
$taskDestination = Join-Path $taskRoot 'frozen_inputs'
if (Test-Path -LiteralPath $taskDestination) { throw 'Refusing an existing frozen input snapshot overwrite' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
Copy-Item -LiteralPath 'C:\Users\LiMing\Documents\改论文\.fresh_wgangp_figures_20261006\external\portable' -Destination (Join-Path $taskDestination 'external_bundle') -Recurse
$taskBaseline = Join-Path $taskDestination 'adopted32_development_baseline'
New-Item -ItemType Directory -Path $taskBaseline | Out-Null
$taskPrevious = 'C:\Users\LiMing\Documents\改论文\.ablation_adopt_and_external_opt_20261006\optimization\HUP065\small_gain_decoupled_screen\results'
Copy-Item -LiteralPath (Join-Path $taskPrevious 'selection_report.json') -Destination (Join-Path $taskBaseline 'selection_report.json')
Copy-Item -LiteralPath (Join-Path $taskPrevious 'selected_pair_adaptation_u150\selected_validation_channel_metrics.csv') -Destination (Join-Path $taskBaseline 'selected_validation_channel_metrics.csv')
Copy-Item -LiteralPath (Join-Path $taskPrevious 'selected_pair_adaptation_u150\selected_validation_trajectory_metrics.csv') -Destination (Join-Path $taskBaseline 'selected_validation_trajectory_metrics.csv')
Copy-Item -LiteralPath 'C:\Users\LiMing\Documents\改论文\.ablation_adopt_and_external_opt_20261006\optimization\HUP065\small_gain_decoupled_screen\original_grid_density_source.csv' -Destination (Join-Path $taskDestination 'original65_density_grid.csv')
Copy-Item -LiteralPath 'C:\Users\LiMing\Documents\改论文\.ablation_adopt_and_external_opt_20261006\optimization\HUP065\small_gain_decoupled_screen\plot_external_layout_snapshot.py' -Destination (Join-Path $taskDestination 'plot_original65_layout_snapshot.py')
Get-ChildItem -LiteralPath $taskDestination -File -Recurse | Measure-Object -Property Length -Sum
