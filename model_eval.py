import torch
import torch.nn as nn

from torchmetrics.functional import peak_signal_noise_ratio


def cals_losses(y_pred, y_true):
    result = {}
    result['mse'] = nn.functional.mse_loss(y_pred, y_true).item()
    result['mae'] = nn.functional.l1_loss(y_pred, y_true).item()
    result['psnr'] = peak_signal_noise_ratio(y_pred, y_true).item()
    return result
    
    
def eval_model_single_frame(model, dataloader, device='cuda'):
    result = {'mse' : 0.0, 'mae' : 0.0, 'psnr' : 0.0}
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            x = batch[0].to(device)
            y_true = batch[1].to(device)

            y_pred = model(x)
            batch_metrics = cals_losses(y_pred, y_true)
            
            for metric in result.keys():
                result[metric] += batch_metrics[metric]
            
        for metric in result.keys():
            result[metric] /= (b_idx + 1)
    return result


def eval_results(dataloader):
    result = {'mse' : 0.0, 'mae' : 0.0, 'psnr' : 0.0}
    with torch.no_grad():
        for (b_idx, batch) in enumerate(dataloader):
            y_true = batch[0]

            y_pred = batch[1]
            elem_metrics = cals_losses(y_pred, y_true)
            
            for metric in result.keys():
                result[metric] += elem_metrics[metric]
            
        for metric in result.keys():
            result[metric] /= (b_idx + 1)
    return result