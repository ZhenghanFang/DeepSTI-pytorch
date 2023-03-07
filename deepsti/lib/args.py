import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSTI for STI Dipole Inversion Problem')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='operation mode: train or predict (default: train)')
    parser.add_argument('--data_dir', type=str, default='data/', help='directory of data')
    parser.add_argument('--name', type=str, default='experiment', help='name of the experiment')
    parser.add_argument('--iter_num', default=4, type=int, help='network parameter, number of unrolled iterations (default: 4)')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'sgdadam'], help='optimizer to use (default: adam)')
    parser.add_argument('--number', type=int, default=6, help='If mode is train, maximum input phase number during training. If mode is predict, input phase number.')
    parser.add_argument('--snr', type=int, default=10, help='phase measurement snr (dB) for training')
    parser.add_argument('--tesla', default=3, type=float, help='B0 tesla (default: 3) in training data')
    parser.add_argument('--use_sampler', type=int, default=2, help='use self-defined sampler for multi-resolution dataset. 1: basic two-stream sampler. 2: weighted two-stream sampler. default: 2')
    parser.add_argument('--train_list', type=str, default='train_input.txt', help='training data list')
    parser.add_argument('--validate_list', type=str, default='validate_input.txt', help='validation data list')
    parser.add_argument('--test_list', type=str, default='test_input.txt', help='test data list')
    parser.add_argument('--model_arch', default='deepsti_resunet', help='network model (default: deepsti_resunet)')
    parser.add_argument('--gpu', type=str, default='0', help='set gpu ids (eg: 0,1)')
    parser.add_argument('--dataset', default='stieff', choices=['stieff'], help='dataset to use (default: stieff)')
    parser.add_argument('--n_w', default=16, type=int, help='number of workers for pytorch dataloader')
    parser.add_argument('--is_aug', type=lambda x: bool(int(x)), default=True, help='Whether to apply data augmentation. 0: False; other: True. default: True.')
    parser.add_argument('--random_nori', type=lambda x: bool(int(x)), default=True, help='randomly downsample number of orientations (choose from 1~6) during training. 0: False; other: True. default: True.')
    parser.add_argument('--num_epoch', default=500, type=int, help='number of epochs to run (default: 500)')
    parser.add_argument('--samples_per_epoch', type=int, default=2000, help='how many samples per epoch')
    
    
    # network arguments
    parser.add_argument('--feat_dim', default=64, type=int, help='network parameter, feature dimension (default: 64)')
    parser.add_argument('--num_blocks', default=8, type=int, help='network parameter, number of ResBlocks (default: 8)')
    
    # optimizer arguments
    parser.add_argument('--batch_size', type=int, default=2, help='batch size (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no_save', action='store_true', default=False, help='disables saving tensors')
    parser.add_argument('--resume_file', type=str, default=None, help='the checkpoint file to resume from')
    
    parser.add_argument('--ext_data', type=str, default='', help='external data info (yml)')
    
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--dk_size', type=int, default=0, help='dipole kernel size')
    
    
    parser.add_argument('--train_step_size', type=lambda x: bool(int(x)), default=True, help='whether to learn the step size in proximal gradient descent. If True, step size is trainable; otherwise, fixed. 0: False; other: True.')
    
    
    
    args = parser.parse_args()
    return args

