"""Replace only the adopted primary controller curve with saved figure data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_data/source_data_control_densities.csv'
path = ROOT / 'data_geometry.json'
geometry = json.loads((ROOT/'source_data/data_geometry.json').read_text(encoding='utf-8'))
table = pd.read_csv(SOURCE)
rows = table[(table['plot'] == 'representative') & (table['node'] == 'RPFa3')]
print('Current source series:', rows['series'].unique().tolist())
aliases = {'Observed ictal': 'Observed ictal', 'Free RC-SDE': 'Free Graph-RC',
           'Interictal reference': 'Preictal reference', 'Actor + WGAN-GP': 'Full WGAN-GP'}
available = rows['series'].unique().tolist()
# The aliases above document historical figure terminology. Resolve the current
# labels from explicit roles; do not reorder or rescale the observations.
expected = [['Observed ictal'], ['Free Graph–RC', 'Free Graph-RC', 'Free RC-SDE', 'Uncontrolled'],
            ['Preictal reference', 'Interictal reference'],
            ['WGAN-GP Full', 'Full WGAN-GP', 'Actor + WGAN-GP', 'Controlled', 'Full']]
xmin, xmax = geometry['densities']['xlim']
ymin, ymax = geometry['densities']['ylim']
checks = []
for index, series in enumerate(geometry['densities']['series']):
    label = next((x for x in expected[index] if x in available), None)
    if label is None:
        raise ValueError((index, available))
    data = rows[rows['series'] == label].sort_values('standardized_amplitude')
    x = data['standardized_amplitude'].to_numpy(float)
    y = data['density'].to_numpy(float)
    assert len(x) == 300
    np.testing.assert_allclose(x, series['x'], rtol=0, atol=1e-14)
    if index < 3:
        np.testing.assert_allclose(y, series['y'], rtol=0, atol=1e-13)
        checks.append({'series': label, 'unchanged_max_error': float(np.max(np.abs(y-series['y'])))})
    else:
        assert y.min() >= ymin and y.max() <= ymax
        checks.append({'series': label, 'maximum_change_from_historical': float(np.max(np.abs(y-series['y'])))})
        series['y'] = y.tolist()
        series['points'] = np.column_stack(((x-xmin)/(xmax-xmin), 1-(y-ymin)/(ymax-ymin))).tolist()
geometry['current_density_source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
geometry['current_primary_actor_sha256'] = '980952488c536b76cfc51fa5d979243e0a69be4cba1cbf36aacdc094e7b29ab0'
geometry['current_density_source'] = str(SOURCE.relative_to(ROOT))
path.write_text(json.dumps(geometry, ensure_ascii=False, allow_nan=False), encoding='utf-8')
(ROOT/'qa/geometry_update_receipt.json').write_text(json.dumps({'checks': checks, 'data_geometry_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'unchanged_layout_and_axis': True, 'no_density_fitting_or_new_rollout': True}, indent=2), encoding='utf-8')
print(json.dumps(checks))
