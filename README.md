
# DeepSTI: Towards Tensor Reconstruction using Fewer Orientations in Susceptibility Tensor Imaging
This is the official implementation of the paper

[DeepSTI: Towards Tensor Reconstruction using Fewer Orientations in Susceptibility Tensor Imaging](https://www.sciencedirect.com/science/article/pii/S1361841523000890). *Medical Image Analysis* 2023.

by [Zhenghan Fang](https://zhenghanfang.github.io/), Kuo-Wei Lai, Peter van Zijl, Xu Li, and [Jeremias Sulam](https://sites.google.com/view/jsulam).

<img src='assets/deepsti_animation_7T_4ori.gif' style="width: 50%">

## Updates
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
```
python deepsti/main.py 

arguments:
--mode                        train (train or predict)
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


### Test on External Data
```
python deepsti/main.py

arguments:
--mode                        predict (train or predict)
--resume_file                 saved model parameters
--ext_data                    yml file of external data information
--gpu                         GPU ID's, e.g. "0" or "0,1"
--output_path                 directory to save predictions
```
Example:
```
python deepsti/main.py --mode predict --resume_file pretrained/deepsti.pkl --gpu 0 --ext_data data/yml/demo.yml --output_path experiment/results
```
Predictions will be saved to `output_path`, with naming convention `[name]_pred_{sti,avg,ani,V1,modpev}.nii.gz`, where `name` is defined in the input yml file. The outputs are:
- `sti`: 6-channel tensor image, ordered as [xx, xy, xz, yy, yz, zz]
- `avg`: mean magnetic susceptibility
- `ani`: magnetic susceptibility anisotropy
- `V1`: principal eigenvector of the susceptibility tensor
- `modpev`: principal eigenvector map modulated by the predicted susceptibility anisotropy

Example outputs from DeepSTI are provided in `results/`.

## Dataset
Demo data for inference is available at `data/test/`. See the "Test on External Data" section for how to run the pretrained model on this example dataset.

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
