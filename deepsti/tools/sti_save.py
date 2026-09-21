
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import math

from lib.StiEvaluationToolkit import StiEvaluationToolkit as stet


def _save_montage(data, out_name, vmin=None, vmax=None):
    slices = range(0, data.shape[2], 10)
    cols = math.ceil(math.sqrt(len(slices)))
    rows = math.ceil(len(slices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, z in zip(axes, slices):
        image = data[:, :, z]
        if image.ndim == 3:
            norm = np.linalg.norm(image, axis=-1)
            scale = np.quantile(norm[norm > 0], 0.95) if np.any(norm > 0) else 1
            image = np.abs(image / np.maximum(norm[..., None], 1e-12) *
                           np.clip(norm[..., None] / scale, 0, 1))
        image = np.transpose(image[::-1, :], (1, 0, 2) if image.ndim == 3 else (1, 0))
        ax.imshow(image, cmap=None if image.ndim == 3 else 'gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'z={z}')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)


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
    _save_montage(ani, out_name + '_ani.png', vmin=0, vmax=0.1)
    print('ani saved.')

    #avg
    avg = avg * mask
    avg_output = nib.Nifti1Image(avg, orig_affine)
    avg_output.to_filename(out_name + '_avg.nii.gz')
    _save_montage(avg, out_name + '_avg.png', vmin=-0.1, vmax=0.1)
    print('avg saved.')

    #V1
    V1 = V1 * mask[:,:,:,None]
    V1_output = nib.Nifti1Image(V1, orig_affine)
    V1_output.to_filename(out_name + '_V1.nii.gz')
    print('V1 saved.')

    # anisotropy-weighted PEV
    modpev = modpev * mask[:,:,:,None]
    modpev_output = nib.Nifti1Image(modpev, orig_affine)
    modpev_output.to_filename(out_name + '_modpev.nii.gz')
    _save_montage(modpev, out_name + '_modpev.png')
    print('modpev saved.')
