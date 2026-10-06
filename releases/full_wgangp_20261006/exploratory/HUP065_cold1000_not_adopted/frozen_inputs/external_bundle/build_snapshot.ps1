# One-time archival assembly. Reproduction uses reproduce_external.py, not this script.
# Copies sealed files as opaque bytes; does not decompress or analyze outer data.
$ErrorActionPreference = 'Stop'
$bundleRoot = $PSScriptRoot
$externalRoot = Split-Path -Parent $bundleRoot
$originalProject = 'D:\VS code\distribution control'
$snapshotProject = Join-Path $bundleRoot 'snapshot\project'
$snapshotPatients = Join-Path $snapshotProject 'patient_results'
$snapshotRunners = Join-Path $bundleRoot 'snapshot\runners'
$bundleFiles = [System.Collections.Generic.List[object]]::new()

function Copy-VerifiedSnapshot {
    param([string]$Original, [string]$Destination, [string]$Role, [string]$Subject = '')
    $originalItem = Get-Item -LiteralPath $Original
    if ($originalItem.PSIsContainer) { throw "Expected a file: $Original" }
    $digest = (Get-FileHash -LiteralPath $Original -Algorithm SHA256).Hash.ToLowerInvariant()
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
            throw "Refusing to overwrite a different existing snapshot: $Destination"
        }
    } else {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
        Copy-Item -LiteralPath $Original -Destination $Destination
    }
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        throw "Copy hash mismatch: $Destination"
    }
    $relative = [System.IO.Path]::GetRelativePath($bundleRoot, $Destination).Replace('\', '/')
    $bundleFiles.Add([ordered]@{relative_path=$relative;original_path=$Original;sha256=$digest;bytes=$originalItem.Length;role=$Role;subject=$Subject})
}

foreach ($sourceFile in Get-ChildItem -LiteralPath (Join-Path $originalProject 'mfc_pipeline') -Filter '*.py' -File) {
    Copy-VerifiedSnapshot $sourceFile.FullName (Join-Path $snapshotProject ('mfc_pipeline\' + $sourceFile.Name)) 'canonical_dependency'
}
Copy-VerifiedSnapshot (Join-Path $originalProject 'part3_mfc\part3_model.py') (Join-Path $snapshotProject 'part3_mfc\part3_model.py') 'canonical_part3'
Copy-VerifiedSnapshot (Join-Path $originalProject 'part2_rc_sde\part2_model.py') (Join-Path $snapshotProject 'part2_rc_sde\part2_model.py') 'canonical_part2'
Copy-VerifiedSnapshot (Join-Path $externalRoot 'run_external_fresh.py') (Join-Path $snapshotRunners 'run_external_fresh.py') 'frozen_runner' 'HUP065'
Copy-VerifiedSnapshot (Join-Path $externalRoot 'run_external_fresh_uniform_reference.py') (Join-Path $snapshotRunners 'run_external_fresh_uniform_reference.py') 'frozen_runner' 'HUP080'

$contractDirectories = @{
    HUP065='runs\HUP065_seed20261011_u1000'
    HUP080='runs\HUP080_seed20261011_u1000_uniformref'
}
foreach ($subject in @('HUP065','HUP080')) {
    $contractPath = Join-Path $externalRoot ($contractDirectories[$subject] + '\training_contract.json')
    $contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
    foreach ($inputProperty in $contract.input_paths.PSObject.Properties) {
        $originalInput = $inputProperty.Name
        if (-not $originalInput.StartsWith((Join-Path $originalProject 'patient_results') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Input is outside the official patient tree: $originalInput"
        }
        if ((Get-FileHash -LiteralPath $originalInput -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputProperty.Value) {
            throw "Input differs from training contract: $originalInput"
        }
        $patientRelative = $originalInput.Substring(((Join-Path $originalProject 'patient_results') + '\').Length)
        Copy-VerifiedSnapshot $originalInput (Join-Path $snapshotPatients $patientRelative) 'official_training_input' $subject
    }
    Copy-VerifiedSnapshot $contractPath (Join-Path $bundleRoot ('snapshot\provenance\' + $subject + '_training_contract.json')) 'training_contract_identity' $subject
}

$sealedFiles = @(
    @{Subject='HUP065';Relative='HUP065_sparse_control_final_v1\frozen_run\science_run\p\OUTER\run03_sealed_segments.npz';Role='sealed_outer_opaque'},
    @{Subject='HUP065';Relative='HUP065_sparse_control_final_v1\frozen_run\science_run\p\OUTER\run03_retrospective_report.json';Role='historical_outer_parity_report_opaque'},
    @{Subject='HUP080';Relative='HUP080_sparse_control_final_v2\frozen_run\science_run\artifacts\15_run04_once\run04_sealed_segments.npz';Role='sealed_outer_opaque'},
    @{Subject='HUP080';Relative='HUP080_sparse_control_final_v2\frozen_run\science_run\artifacts\15_run04_once\run04_retrospective_report.json';Role='historical_outer_parity_report_opaque'}
)
foreach ($sealed in $sealedFiles) {
    Copy-VerifiedSnapshot (Join-Path $originalProject ('patient_results\' + $sealed.Relative)) (Join-Path $snapshotPatients $sealed.Relative) $sealed.Role $sealed.Subject
}
Copy-VerifiedSnapshot (Join-Path $bundleRoot 'reproduce_external.py') (Join-Path $bundleRoot 'reproduce_external.py') 'portable_routing_wrapper'
Copy-VerifiedSnapshot (Join-Path $bundleRoot 'requirements.txt') (Join-Path $bundleRoot 'requirements.txt') 'observed_environment'

$bundleManifest = [ordered]@{
    schema_version=1
    scope='Frozen predictive inputs and exact implementation snapshots for post-hoc fresh-controller reproduction; not adopted external paper results.'
    outer_arrays_unpacked_during_assembly=$false
    new_HUP080_run_results_read_during_assembly=$false
    official_training_input_keys_preserved_as_identity_aliases=$true
    runners=@{HUP065='snapshot/runners/run_external_fresh.py';HUP080='snapshot/runners/run_external_fresh_uniform_reference.py'}
    files=@($bundleFiles | Sort-Object relative_path)
}
$bundleManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $bundleRoot 'bundle_manifest.json') -Encoding utf8
[pscustomobject]@{Files=$bundleFiles.Count;TotalBytes=($bundleFiles | Measure-Object bytes -Sum).Sum;SealedFiles=4;OuterUnpacked=$false}
