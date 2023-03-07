
import nibabel as nib
import numpy as np
import os

from lib.StiEvaluationToolkit import StiEvaluationToolkit as stet

def sti_save(sti_data, original_nifti, mask, out_name='test_output'):
    """
    mask: (w,h,d)
    """

    print(original_nifti)
    orig_nii = nib.load(original_nifti)
    orig_affine = orig_nii.affine

    mask = mask.astype('int')
    sti_data = sti_data * mask[:,:,:,None]
    L, V, avg, ani, V1, modpev = stet.tensor2misc(sti_data)

    #sti
    sti_output = nib.Nifti1Image(sti_data, orig_affine)
    sti_output.to_filename(out_name + '_sti.nii.gz')
    print('STI saved.')

    #ani
    ani = ani * mask
    ani_output = nib.Nifti1Image(ani, orig_affine)
    ani_output.to_filename(out_name + '_ani.nii.gz')
    print('ani saved.')

    #avg
    avg = avg * mask
    avg_output = nib.Nifti1Image(avg, orig_affine)
    avg_output.to_filename(out_name + '_avg.nii.gz')
    print('avg saved.')

    #V1
    V1 = V1 * mask[:,:,:,None]
    V1_output = nib.Nifti1Image(V1, orig_affine)
    V1_output.to_filename(out_name + '_V1.nii.gz')
    print('V1 saved.')

    # MSA-weighted PEV
    modpev = modpev * mask[:,:,:,None]
    modpev_output = nib.Nifti1Image(modpev, orig_affine)
    modpev_output.to_filename(out_name + '_modpev.nii.gz')
    print('modpev saved.')

