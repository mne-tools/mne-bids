import os.path as op
import numpy as np
import json

import nibabel as nib
import mne_bids
import mne

template_dir = '/Users/alexrockhill/Downloads/templates'

templates = mne_bids.config.BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS

for template in templates:
    # load T1, get MGH header
    T1 = nib.load(op.join(template_dir, f'space-{template}_T1w.nii.gz'))
    T1_mgh = nib.MGHImage(np.array(T1.dataobj).astype(np.float32), T1.affine)
    # load json for anatomical landmarks
    with open(op.join(template_dir, f'space-{template}_T1w.json'), 'r') as fid:
        fids = json.loads(fid.read())['AnatomicalLandmarkCoordinates']
    # make montage with fiducials
    montage = mne.channels.make_dig_montage(
        nasion=fids['NAS'], lpa=fids['LPA'], rpa=fids['RPA'],
        coord_frame='mri_voxel')
    vox_mri_trans = mne.transforms.Transform(
        fro='mri_voxel', to='mri', trans=T1_mgh.header.get_vox2ras_tkr())
    # always the same
    vox_mri_trans.save(op.join('mne_bids', 'data',
                               f'space-{template}_vox-mri_trans.fif'))
    montage.apply_trans(vox_mri_trans)
    for d in montage.dig:
        d['r'] /= 1000  # mm -> m
    # save fiducials
    mne.io.write_fiducials(op.join(
        'mne_bids', 'data', f'space-{template}_fiducials.fif'),
        montage.dig, coord_frame='mri')
    # save head -> mri trans
    trans = mne.transforms.invert_transform(
        mne.channels.compute_native_head_t(montage))
    trans.save(op.join(
        'mne_bids', 'data', f'space-{template}_trans.fif'))
    # save ras -> vox
    ras_vox_trans = mne.transforms.Transform(fro='mri_voxel', to='ras',
                                             trans=T1_mgh.header.get_ras2vox())
    ras_vox_trans.save(op.join('mne_bids', 'data',
                               f'space-{template}_ras-vox_trans.fif'))
