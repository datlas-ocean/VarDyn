import os

import pytest

from src.background_paths import background_control_path


def test_background_control_path_uses_sibling_experiment(tmp_path):
    current = tmp_path / 'current' / 'controls' / 'analysis'
    subwindow = 'subwindow_2025-01-15/tile_0'

    path = background_control_path(current, 'reference', subwindow)

    assert path == os.path.join(
        tmp_path, 'current', 'controls', 'reference', subwindow, 'Xres.nc')


@pytest.mark.parametrize('control_root,experiment_name', [
    (None, 'reference'),
    ('/controls/analysis', None),
])
def test_background_control_path_requires_root_and_source(
        control_root, experiment_name):
    with pytest.raises(ValueError, match='Background mode requires'):
        background_control_path(control_root, experiment_name, 'window/tile')
