
# DeepSTI: Towards Tensor Reconstruction using Fewer Orientations in Susceptibility Tensor Imaging
This is the official implementation of the paper

[DeepSTI: Towards Tensor Reconstruction using Fewer Orientations in Susceptibility Tensor Imaging](https://www.sciencedirect.com/science/article/pii/S1361841523000890). *Medical Image Analysis* 2023.

by [Zhenghan Fang](https://zhenghanfang.github.io/), [Kuo-Wei Lai](https://kuoweilai.com/), [Peter van Zijl](https://profiles.hopkinsmedicine.org/provider/peter-c-van-zijl/2777129), [Xu Li](https://profiles.hopkinsmedicine.org/provider/xu-li/2777501), and [Jeremias Sulam](https://jsulam.github.io/).

<img src='assets/deepsti_animation_7T_4ori.gif' style="width: 50%">

## Updates
- 2026-09-20: Add example training data, output montages, and the demo DTI reference.
- 2026-03-31: Add pretrained checkpoint and demo data for testing.

## Requirements
- [Python 3.9](https://www.python.org/)
- [PyTorch 1.12.0](https://pytorch.org)

## Environment Settings
Use the command below to install all required libraries.
```
conda env create --name [MY_ENV] -f environment.yml
```

## Usage
Activate conda environment first
```
conda activate [MY_ENV]
```

### Train

Download the [DeepSTI synthetic training dataset](https://huggingface.co/datasets/ZhenghanFang/DeepSTI-training-data) from Hugging Face:

```bash
python -m pip install -U huggingface_hub

hf download ZhenghanFang/DeepSTI-training-data \
  --repo-type dataset \
  --local-dir data/synthetic
```

The download already contains the training patches. They can also be recreated
from the included whole-image arrays (the command skips existing patches):

```bash
python scripts/generate_training_patches.py --data_dir data/synthetic
```

Train DeepSTI with:

```bash
python deepsti/main.py \
  --mode train \
  --name default \
  --data_dir data/synthetic \
  --gpu 0
```

The dataset uses `Sub001`, `Sub002`, `Sub007`, `Sub008`, and `Sub009` for
training, `Sub005` for validation, and `Sub003` and `Sub006` for testing.

For a custom dataset, the training options are:

```
python deepsti/main.py --mode train

arguments:
--name                        name of your experiment
--data_dir                    path to dataset directory
--train_list                  list of training data
--validate_list               list of validation data
--test_list                   list of testing data
--tesla                       field strength in training data [default: 3]
--batch_size                  batch size [default is 2]
--gpu                         GPU ID's, e.g. "0" or "0,1"
```
Example:
```
python deepsti/main.py --mode train --name myexp --data_dir data/ --train_list train.txt --validate_list validate.txt --test_list test.txt --gpu 0,1
```
#### Tensorboard Visualization
```
tensorboard --logdir experiment/tb_log/deepsti_resunet_myexp
```

### Pretrained Checkpoint

Download the pretrained checkpoint [`deepsti.pkl` here](https://huggingface.co/ZhenghanFang/DeepSTI-pytorch/tree/main) and save it as `pretrained/deepsti.pkl`.


### Test on Your Own Data
```
python deepsti/main.py --mode predict

arguments:
--resume_file                 saved model parameters
--ext_data                    yml file of external data information
--gpu                         GPU ID's, e.g. "0" or "0,1"
--output_path                 directory to save predictions
```
Example:
```
python deepsti/main.py --mode predict --resume_file pretrained/deepsti.pkl --gpu 0 --ext_data data/yml/demo.yml --output_path experiment/results
```
Predictions will be saved to `output_path`, with `name` in the input yml file as prefix. The outputs are:
- `sti`: 6-channel tensor image, ordered as [xx, xy, xz, yy, yz, zz]
- `avg`: mean magnetic susceptibility
- `ani`: magnetic susceptibility anisotropy
- `V1`: principal eigenvector of the susceptibility tensor
- `modpev`: principal eigenvector map modulated by the predicted susceptibility anisotropy

Mean susceptibility, anisotropy, and modPEV are saved as both NIfTI files
and montage PNGs in the same output folder.

Example outputs from DeepSTI are provided in `results/`.

## Dataset
The synthetic training, validation, and test data are available in the
[DeepSTI training-data repository on Hugging Face](https://huggingface.co/datasets/ZhenghanFang/DeepSTI-training-data).
The release keeps the NumPy directory layout expected by the data loader.

Demo data for inference is available at `data/test/`, along with the registered DTI
reference.

To prepare your own data for inference, 
- use LPS+ orientation for the frequency map, mask, and B0 direction
- set voxel size correctly in the NIfTI header, as it will be read from the image metadata via `nib.load(...).header.get_zooms()`


## References

If you find the code useful for your research, please consider citing
```bib
@article{fang2023deepsti,
  title={Deepsti: towards tensor reconstruction using fewer orientations in susceptibility tensor imaging},
  author={Fang, Zhenghan and Lai, Kuo-Wei and van Zijl, Peter and Li, Xu and Sulam, Jeremias},
  journal={Medical image analysis},
  volume={87},
  pages={102829},
  year={2023},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.media.2023.102829}
}
```
