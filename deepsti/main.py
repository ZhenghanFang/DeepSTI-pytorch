import os
import sys
import argparse
from lib.args import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils import data

from lib.utils import prepareDataset, loadData, chooseModel, chooseLoss, chooseOptimizer
from tools.sti_save import sti_save
from tools.code_snapshot import code_snapshot

from lib.StiEvaluationToolkit import StiEvaluationToolkit as stet

import timeit
import nibabel as nib
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
data_normalization = True
record_time = False


data_root = args.data_dir

npy_dir = data_root + '/sti_sub/'
nifti_dir = data_root + '/sti_data/'

checkpoint_dir = './experiment/checkpoint/' # models
tb_log_dir = './experiment/tb_log/' # tensorboard
vis_output_path = './experiment/results/' # reconstruction results
snapshot_dir = './experiment/code_snapshots/' # code snapshots

## load max/min value of sti data computed from training dataset. Useful for computing psnr and ssim.
# max/min over all 6 channels
data_max = np.load('./data/stats/train_val_gt_max_val.npy')
data_min = np.load('./data/stats/train_val_gt_min_val.npy')
# max/min per channel
data_max_vec = np.load('./data/stats/train_val_gt_max_vec.npy')
data_min_vec = np.load('./data/stats/train_val_gt_min_vec.npy')



def main(args):

    start = timeit.default_timer()
    device = torch.device('cuda' if not args.no_cuda else 'cpu')
    
    # prediction data number
    if args.ext_data == '':
        prediction_set = 'test' #['train', 'val', 'test', 'ext']
    else:
        prediction_set = 'ext'
    case = 'whole' #['patch', 'whole']
    prediction_data = (prediction_set, None, None, None, case)

    if args.mode == 'train':
        ## experiment name
        global exp_name
        exp_name = args.model_arch + '_' + args.name
        
        code_snapshot(os.path.join(snapshot_dir, exp_name), sys.argv)

        ## tensorboard log
        tb_writer = SummaryWriter(tb_log_dir + exp_name)

        ## load dataset
        print('Load dataset...')
        train_loader, val_loader, test_loader = prepareDataset(args, device, npy_dir, data_normalization, args.n_w)
        print(args.dataset.lower() + ' dataset loaded.')

        ## load model
        model = chooseModel(args)
        model.to(device)
        print(args.model_arch + ' loaded.')

        ## parallel model
        model = nn.DataParallel(model)

        ## loss function and optimizer
        loss_fn = chooseLoss(args, 1)
        optimizer = chooseOptimizer(model, args)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

        ## initailize statistic result
        start_epoch = 0
        best_mse_index = 1000
        best_vec_index = 1000
        best_wpsnr_index = -1000
        total_tb_it = 0

        ## load pretrained model
        if not args.resume_file == None:
            state = torch.load(args.resume_file, map_location=device)
            model.load_state_dict(state['model_state'])
            model.to(device)
            print(args.resume_file + ' loaded.')

            
        for epoch in range(start_epoch, args.num_epoch):

            print(epoch, optimizer)
            for param_group in optimizer.param_groups:
                print('Learning rate: %.8f' %(param_group['lr']))
            tb_writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'])
            
            total_tb_it = train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it)
            mse_index, vec_index, wpsnr_index = validate(device, model, val_loader, epoch, loss_fn, tb_writer, split='val')
            _, _, _ = validate(device, model, test_loader, epoch, loss_fn, tb_writer, split='test')

            state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 
                     'best_mse_index': best_mse_index, 'best_vec_index': best_vec_index}
            
            os.makedirs(checkpoint_dir, exist_ok=True)
#             if epoch % 100 == 0 and epoch > 0:
#                 name = checkpoint_dir + exp_name +'_Ep{}.pkl'.format(epoch)
#                 torch.save(state, name)
            
            scheduler.step()
            
            if mse_index <= best_mse_index:
                best_mse_index = mse_index
                best_name = checkpoint_dir + exp_name +'_Mmodel.pkl'
                torch.save(state, best_name)

            if vec_index <= best_vec_index:
                best_vec_index = vec_index
                best_name = checkpoint_dir + exp_name +'_Vmodel.pkl'
                torch.save(state, best_name)
                
            if wpsnr_index >= best_wpsnr_index:
                best_wpsnr_index = wpsnr_index
                best_name = checkpoint_dir + exp_name +'_Wmodel.pkl'
                torch.save(state, best_name)

            name = checkpoint_dir + exp_name +'_Emodel.pkl'
            torch.save(state, name)
            
        tb_writer.close()

    elif args.mode == 'predict':

        print('Load data...')
        data_loader = loadData(args, device, npy_dir, prediction_data, data_normalization)

        ## load model
        model = chooseModel(args)
        
        ## load gt_mean, gt_std
        if data_normalization and len(args.ext_data) > 0:
            data_loader.dataset.set_mean_std(model.gt_mean.squeeze().numpy(), model.gt_std.squeeze().numpy())

        ## parallel model
        model = nn.DataParallel(model)

        if not args.resume_file == None:
            model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])

        model.to(device)
        print(sum([p.numel() for p in model.parameters()]))
        print(args.model_arch + ' loaded.')

        if not args.resume_file == None:
            model_name = args.resume_file.split('/')[-1].split('.')[0]
        else:
            model_name = 'no_train'
        print(model_name)

        predict(args, device, model, data_loader, model_name, prediction_data)
    else:
        raise Exception('Unrecognized mode.')
    stop = timeit.default_timer()
    print('Time: ', stop - start)

def train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it):

    print_freq = (len(train_loader.dataset) // args.batch_size) // 300

    model.train()

    print(len(train_loader))
    for batch_count, (input_data_list, gt_data, mask_data_list, dk_data_list, ani_mask, _, sub_name) in enumerate(tqdm(train_loader, ncols=60)):

        #cuda
        input_data = input_data_list.to(device, dtype=torch.float)
        gt_data = gt_data.to(device, dtype=torch.float)
        mask_data = mask_data_list.to(device, dtype=torch.float)
        dk_data = dk_data_list.to(device, dtype=torch.float)
        ani_mask = ani_mask.to(device, dtype=torch.float)
        
        optimizer.zero_grad()

        
        output_data = model(input_data, dk_data, mask_data.unsqueeze(1), None)
        
        
        loss = loss_fn(output_data, gt_data)

        loss.backward()

        optimizer.step()
        

        per_loss = loss.item()

        tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
        tb_writer.add_scalar('train/alpha', model.module.alpha, total_tb_it)
        total_tb_it += 1

        if batch_count%print_freq == 0:
            print('Epoch [%d/%d] Loss: %.8f' %(epoch, args.num_epoch, per_loss))
            
            
        if batch_count*args.batch_size >= args.samples_per_epoch:
            break

    return total_tb_it

def validate(device, model, val_loader, epoch, loss_fn, tb_writer, split):

    model.eval()

    tb_loss = 0
    mse_loss = 0
    vec_perf = 0
    ssim_perf = 0
    psnr_perf = 0
    wpsnr_perf = 0

    with torch.no_grad():

        for batch_count, (input_data_list, gt_data, mask_data_list, dk_data_list, ani_mask, _, sub_name) in enumerate(tqdm(val_loader, ncols=50)):
            
            batch_size = input_data_list.shape[0]
            
            #cuda
            input_data = input_data_list.to(device, dtype=torch.float)
            gt_data = gt_data.to(device, dtype=torch.float)
            mask_data = mask_data_list.to(device, dtype=torch.float)
            dk_data = dk_data_list.to(device, dtype=torch.float)
            ani_mask = ani_mask.to(device, dtype=torch.float)


            output_data = model(input_data, dk_data, mask_data.unsqueeze(1), None)

            loss = loss_fn(output_data, gt_data)

            tb_loss += loss.item() * batch_size

            mask = torch.squeeze(mask_data[:, 0, :, :, :], 0).cpu().numpy()
            ani_m = torch.squeeze(ani_mask[:, :, :, :], 0).cpu().numpy()

            if data_normalization:
                og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
                og_output = og_output + val_loader.dataset.gt_mean
                og_output = og_output * mask[:,:,:,np.newaxis]

                og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
                og_gt = og_gt + val_loader.dataset.gt_mean
                og_gt = og_gt * mask[:,:,:,np.newaxis]

            else:
                og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask[:,:,:,np.newaxis]
                og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask[:,:,:,np.newaxis]

            print('computing metrics...')
            mse_val = stet.mse_sti(og_output, og_gt, mask)
            mse_loss += mse_val
            psnr_val = stet.psnr_sti(og_output, og_gt, mask, data_min, data_max)
            psnr_perf += psnr_val
            ssim_val = stet.ssim_sti(og_gt, og_output, data_max_vec, data_min_vec)
            ssim_perf += ssim_val

            gt_L, gt_V, gt_avg, gt_ani, gt_V1, gt_modpev = stet.tensor2misc(og_gt)
            pred_L, pred_V, pred_avg, pred_ani, pred_V1, pred_modpev = stet.tensor2misc(og_output)
            vec_val = stet.evec_cos_sim_err(gt_V1, pred_V1, mask, ani_m, ani_thr=0.015)
            vec_perf += vec_val
            wpsnr_tmp = stet.wpsnr_sti(pred_V1, pred_ani, gt_V1, gt_ani, mask)
            wpsnr_perf += wpsnr_tmp

        avg_tb_loss = tb_loss / len(val_loader.dataset)
        avg_mse_loss = mse_loss / len(val_loader.dataset)
        avg_vec_perf = vec_perf / len(val_loader.dataset)
        avg_psnr_perf = psnr_perf / len(val_loader.dataset)
        avg_ssim_perf = ssim_perf / len(val_loader.dataset)
        avg_wpsnr_perf = wpsnr_perf / len(val_loader.dataset)
        
        #print('alpha: %.3f' %(model.alpha.cpu().numpy()))
        print('##', split, 'loss: %.8f Mse: %.8f PSNR: %.8f SSIM: %.8f VEC: %.8f WPSNR: %.8f' %(avg_tb_loss, avg_mse_loss, avg_psnr_perf, avg_ssim_perf, avg_vec_perf, avg_wpsnr_perf))
        

        tb_writer.add_scalar(f'{split}/overall_loss', avg_tb_loss, epoch)
        tb_writer.add_scalar(f'{split}/Mse', avg_mse_loss, epoch)
        tb_writer.add_scalar(f'{split}/PSNR', avg_psnr_perf, epoch)
        tb_writer.add_scalar(f'{split}/SSIM', avg_ssim_perf, epoch)
        tb_writer.add_scalar(f'{split}/VEC', avg_vec_perf, epoch)
        tb_writer.add_scalar(f'{split}/WPSNR', avg_wpsnr_perf, epoch)

    return avg_mse_loss, avg_vec_perf, avg_wpsnr_perf




def predict(args, device, model, data_loader, model_name, prediction_data):
    
    prediction_set, subject_num, ori_num, patch_num, case = prediction_data
    
    model.eval()

    mse_loss = 0
    ssim_perf = 0
    psnr_perf = 0
    vec_perf = 0
    vec_thr0_perf = 0
    wpsnr_perf = 0

    with torch.no_grad():
        for batch_count, (input_data_list, gt_data, mask_data_list, dk_data_list, ani_mask, idx, sub_name) in enumerate(tqdm(data_loader)):

            subject = sub_name[0][:6]

            #cuda
            input_data = input_data_list.to(device, dtype=torch.float)
            gt_data = gt_data.to(device, dtype=torch.float)
            mask_data = mask_data_list.to(device, dtype=torch.float)
            dk_data = dk_data_list.to(device, dtype=torch.float)
            ani_mask = ani_mask.to(device, dtype=torch.float)

            print(next(model.module.parameters()).device, input_data.device)
            if not next(model.module.parameters()).is_cuda:
                model = model.module
            output_data = model(input_data, dk_data, mask_data.unsqueeze(1), None)
            
            mask = torch.squeeze(mask_data[:, 0, :, :, :], 0).cpu().numpy()
            ani_m = torch.squeeze(ani_mask[:, :, :, :], 0).cpu().numpy()

            if data_normalization:
                og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
                og_output = og_output + data_loader.dataset.gt_mean
                og_output = og_output * mask[:,:,:,np.newaxis]

                og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
                og_gt = og_gt + data_loader.dataset.gt_mean
                og_gt = og_gt * mask[:,:,:,np.newaxis]

            else:
                og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask[:,:,:,np.newaxis]
                og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask[:,:,:,np.newaxis]

            print('computing metrics...')
            mse_val = stet.mse_sti(og_output, og_gt, mask)
            mse_loss += mse_val
            psnr_val = stet.psnr_sti(og_output, og_gt, mask, data_min, data_max)
            psnr_perf += psnr_val
            ssim_val = stet.ssim_sti(og_gt, og_output, data_max_vec, data_min_vec)
            ssim_perf += ssim_val

            gt_L, gt_V, gt_avg, gt_ani, gt_V1, gt_modpev = stet.tensor2misc(og_gt)
            pred_L, pred_V, pred_avg, pred_ani, pred_V1, pred_modpev = stet.tensor2misc(og_output)
            vec_thr0_val = stet.evec_cos_sim_err(gt_V1, pred_V1, mask, ani_m, ani_thr=0.0)
            vec_thr0_perf += vec_thr0_val
            vec_val = stet.evec_cos_sim_err(gt_V1, pred_V1, mask, ani_m, ani_thr=0.015)
            vec_perf += vec_val
            wpsnr_tmp = stet.wpsnr_sti(pred_V1, pred_ani, gt_V1, gt_ani, mask)
            wpsnr_perf += wpsnr_tmp
            
            print(sub_name[0])
            print('##Test PSNR: %.8f SSIM: %.8f VEC: %.8f VEC(thr0): %.4f WPSNR: %.4f MSE: %.4f' %(psnr_val, ssim_val, vec_val, vec_thr0_val, wpsnr_tmp, mse_val))

            if not args.no_save:
                if prediction_set == 'ext':
                    phase_fn = data_loader.dataset.data_info[idx]['phase'][0]
                    if phase_fn.endswith('.nii.gz'):
                        orig_nii_path = phase_fn
                    else:
                        orig_nii_path = data_loader.dataset.data_info[idx]['nii']
                        # orig_nii_path = phase_fn.replace('.npy', '.nii.gz').replace('sti_sub/whole', 'sti_data')
                else:
                    orig_nii_path = nifti_dir + '/' + subject + '/sti/' + subject + '_sti_tensor.nii.gz'
                    
                if not os.path.exists(vis_output_path + model_name):
                    os.makedirs(vis_output_path + model_name)

                if prediction_set == 'ext':
                    save_name = vis_output_path + model_name + '/' + sub_name[0] + '_pred'
                else:
                    save_name = vis_output_path + model_name + '/' + sub_name[0] + '_snr{}'.format(data_loader.dataset.snr) + '_pred'

                sti_save(og_output, orig_nii_path, mask, out_name=save_name)
                
        avg_mse_loss = mse_loss / len(data_loader.dataset)
        avg_psnr_perf = psnr_perf / len(data_loader.dataset)
        avg_ssim_perf = ssim_perf / len(data_loader.dataset)
        avg_vec_perf = vec_perf / len(data_loader.dataset)
        avg_vec_thr0_perf = vec_thr0_perf / len(data_loader.dataset)
        avg_wpsnr_perf = wpsnr_perf / len(data_loader.dataset)

        print('##Test Mse: %.8f PSNR: %.8f SSIM: %.8f VEC: %.8f VEC(thr0): %.4f WPSNR: %.8f' %(avg_mse_loss, avg_psnr_perf, avg_ssim_perf, avg_vec_perf, avg_vec_thr0_perf, avg_wpsnr_perf))


if __name__ == '__main__':
    main(args)
