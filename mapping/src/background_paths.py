"""Paths for control vectors used by windowed assimilation."""

import os


def background_control_path(control_root, experiment_name, name_subwindow):
    """Find a source experiment's tile control beside the current experiment."""
    if control_root is None:
        raise ValueError(
            'Background mode requires INV.path_save_control_vectors to point '
            'to the current experiment control directory.')
    if not experiment_name:
        raise ValueError(
            'Background mode requires name_exp_background (or name_exp).')
    control_parent = os.path.dirname(os.path.normpath(os.fspath(control_root)))
    return os.path.join(control_parent, experiment_name, name_subwindow,
                        'Xres.nc')
