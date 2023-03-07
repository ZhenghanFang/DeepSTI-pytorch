
# DeepSTI
[Arxiv](https://arxiv.org/abs/2209.04504 "Arxiv")

### Official PyTorch implementation of the paper "DeepSTI: Towards Tensor Reconstruction using Fewer Orientations in Susceptibility Tensor Imaging" [Arxiv](https://arxiv.org/abs/2209.04504 "Arxiv")

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
--data                        path to dataset directory
--train_list                  list of training data
--validate_list               list of validation data
--test_list                   list of testing data
--tesla                       field strength in training data [default: 3]
--batch_size                  batch size [default is 2]
--gpu                         GPU ID's, e.g. "0" or "0,1"
```
Example:
```
python deepsti/main.py --mode train --name myexp --data data/ --train_list train.txt --validate_list validate.txt --test_list test.txt --gpu 0,1
```
#### Tensorboard Visualization
```
tensorboard --logdir experiment/tb_log/deepsti_resunet_myexp
```

### Test on External Data
```
python deepsti/main.py

arguments:
--mode                        predict (train or predict)
--resume_file                 saved model parameters
--ext_data                    yml file of external data information
--gpu                         GPU ID's, e.g. "0" or "0,1"
```
Example:
```
python deepsti/main.py --mode predict --resume_file experiment/checkpoint/deepsti_resunet_Vmodel.pkl --gpu 1 --ext_data data/yml/example.yml
```
Predictions will be saved in ```experiment/results```.

### Dataset
Demo data will be provided shortly.


