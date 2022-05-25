import torch
import torch.nn as nn
import numpy as np

from torchmetrics.functional import peak_signal_noise_ratio


def cals_losses(y_pred, y_true):
    result = {}
    result['mse'] = nn.functional.mse_loss(y_pred, y_true).item()
    result['mae'] = nn.functional.l1_loss(y_pred, y_true).item()
    result['psnr'] = peak_signal_noise_ratio(y_pred, y_true, data_range=1.0).item()
    return result
    
    
def eval_model_single_frame(model, dataloader, device='cuda', verbose=False):
    result = {'mse' : [], 'mae' : [], 'psnr' : []}
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            x = batch[0].to(device)
            y_true = batch[1].to(device)

            y_pred = model(x)
            if isinstance(y_pred, list):
                y_pred = y_pred[-1]

            batch_metrics = cals_losses(y_pred, y_true)
            
            for metric in result.keys():
                result[metric].append(batch_metrics[metric])
        if verbose:
            print('Evaluation completed. Metrics: ')
            for metric in result.keys():
                print(metric, ": ", np.mean(result[metric]), ' +- ', np.std(result[metric]))
        else:
            return {'mse': np.mean(result['mse']), 'psnr': np.mean(result['psnr']), 'mae': np.mean(result['mae'])}
        
def eval_model_multiframe(model, dataloader, device='cuda', frames=4, verbose=False):
    result = {'mse' : [], 'mae' : [], 'psnr' : []}
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            x = batch[0].to(device)
            y_true = batch[1].to(device)

            y_pred = model(x)
            batch_metrics = {'mse' : 0.0, 'mae' : 0.0, 'psnr' : 0.0}
            for t_i in range(frames - 1, frames):
                batch_metrics_partial = cals_losses(y_pred[t_i], y_true[t_i : len(y_true) - frames + 1 + t_i ]) 
                for key in batch_metrics_partial.keys():
                    batch_metrics[key] = batch_metrics_partial[key] #/ frames
            
            for metric in result.keys():
                result[metric].append(batch_metrics[metric])
        if verbose:
            print('Evaluation completed. Metrics: ')
            for metric in result.keys():
                print(metric, ": ", np.mean(result[metric]), ' +- ', np.std(result[metric]))
        else:
            return {'mse': np.mean(result['mse']), 'psnr': np.mean(result['psnr']), 'mae': np.mean(result['mae'])}


def eval_results(dataloader):
    result = {'mse' :[], 'mae' : [], 'psnr' : []}
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            y_true = batch[0]

            y_pred = batch[1]
            elem_metrics = cals_losses(y_pred, y_true)
            
            for metric in result.keys():
                result[metric].append(batch_metrics[metric])
        print('Evaluation completed. Metrics: ')
        for metric in result.keys():
            print(metric, ": ", np.mean(result[metric]), ' +- ', np.std(result[metric]))
    return result
