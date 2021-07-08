import json

import numpy as np
from scipy import linalg
import nibabel as nib

import mne


def plot_anat_landmarks(bids_path, vmax=None, show=True):
    """Plot anatomical landmarks attached to an MRI image.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path of the MRI image.
    vmax : float
        Maximum colormap value.
    show : bool
        Whether to show the figure after plotting. Defaults to ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_anat

    nii = nib.load(str(bids_path))

    json_path = bids_path.copy().update(extension=".json")

    n_landmarks = 0
    if json_path.fpath.exists():
        json_content = json.load(open(json_path))
        coords_dict = json_content.get("AnatomicalLandmarkCoordinates", dict())
        n_landmarks = len(coords_dict)

    if not n_landmarks:
        raise ValueError("No landmarks available with the image")

    for label in coords_dict:
        vox_pos = np.array(coords_dict[label])
        ras_pos = mne.transforms.apply_trans(nii.affine, vox_pos)
        coords_dict[label] = ras_pos

    ########################################################################
    # Plot it with nilearn
    fig, axs = plt.subplots(
        n_landmarks, 1, figsize=(6, 2.3 * n_landmarks),
        facecolor="w")

    for point_idx, (label, ras_pos) in enumerate(coords_dict.items()):
        plot_anat(
            str(bids_path), axes=axs[point_idx], cut_coords=ras_pos,
            title=label, vmax=vmax,
        )

    plt.suptitle(bids_path.fpath.name)

    if show:
        fig.show()

    return fig
