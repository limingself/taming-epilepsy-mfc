$ErrorActionPreference = 'Stop'
$taskSource = 'C:\Users\LiMing\Documents\改论文\.fresh_wgangp_figures_20261006\external\portable'
$taskFolder = 'C:\Users\LiMing\Documents\改论文\.ablation_adopt_and_external_opt_20261006\optimization\HUP065\small_gain_decoupled_screen'
$taskTarget = Join-Path $taskFolder 'portable_inputs'
if (Test-Path -LiteralPath $taskTarget) { throw 'Refusing existing portable snapshot overwrite' }
New-Item -ItemType Directory -Path $taskTarget | Out-Null
Copy-Item -LiteralPath $taskSource -Destination (Join-Path $taskTarget 'external_bundle') -Recurse
$taskOldSource = 'C:\Users\LiMing\Documents\改论文\.fresh_wgangp_figures_20261006\external\runs\HUP065_seed20261011_u1000\frozen_actor_wgan.pt'
$taskOldFolder = Join-Path $taskTarget 'old23_source_run'
New-Item -ItemType Directory -Path $taskOldFolder | Out-Null
Copy-Item -LiteralPath $taskOldSource -Destination (Join-Path $taskOldFolder 'frozen_actor_wgan.pt')
$taskSourcesFolder = Join-Path $taskTarget 'optimizer_sources'
New-Item -ItemType Directory -Path $taskSourcesFolder | Out-Null
Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $taskFolder) 'optimize_hup065.py') -Destination (Join-Path $taskSourcesFolder 'optimize_hup065.py')
Copy-Item -LiteralPath (Join-Path $taskFolder 'screen_small_gain.py') -Destination (Join-Path $taskSourcesFolder 'screen_small_gain.py')
Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $taskFolder) 'finalize_hup065.py') -Destination (Join-Path $taskSourcesFolder 'finalize_hup065.py')
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskOldFolder 'frozen_actor_wgan.pt'), (Join-Path $taskSourcesFolder 'optimize_hup065.py'), (Join-Path $taskSourcesFolder 'screen_small_gain.py'), (Join-Path $taskSourcesFolder 'finalize_hup065.py') | Select-Object Path,Hash
