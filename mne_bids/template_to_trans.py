import os.path as op
import numpy as np
import json

import nibabel as nib
import mne_bids
import mne

template_dir = '/Users/alexrockhill/Downloads/templates'

templates = mne_bids.config.BIDS_STANDARD_TEMPLATE_COORDINATE_FRAMES

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
    # get transforms, x->head and head->mri, first vox->head
    vox_head_t = mne.channels.compute_native_head_t(montage)
    vox_head_t.save(op.join(  # note: still needs mm->m
        'mne_bids', 'data', f'space-{template}_fro-vox_to-head_trans.fif'))
    # get ras->head
    vox_ras_trans = mne.transforms.Transform(fro='mri_voxel', to='ras',
                                             trans=T1_mgh.header.get_vox2ras())
    montage.apply_trans(vox_ras_trans)
    ras_head_t = mne.channels.compute_native_head_t(montage)
    ras_head_t.save(op.join(  # note: still needs mm->m
        'mne_bids', 'data', f'space-{template}_fro-ras_to-head_trans.fif'))
    # reset montage, go to surface RAS this time
    montage = mne.channels.make_dig_montage(
        nasion=fids['NAS'], lpa=fids['LPA'], rpa=fids['RPA'],
        coord_frame='mri_voxel')
    vox_mri_trans = mne.transforms.Transform(
        fro='mri_voxel', to='mri', trans=T1_mgh.header.get_vox2ras_tkr())
    montage.apply_trans(vox_mri_trans)
    mri_head_t = mne.channels.compute_native_head_t(montage)
    head_mri_t = mne.transforms.invert_transform(mri_head_t)
    head_mri_t.save(op.join(  # note: still needs mm->m
        'mne_bids', 'data', f'space-{template}_fro-head_to-mri_trans.fif'))
