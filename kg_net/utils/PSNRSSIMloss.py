from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import scipy.io as scio
import os
from typing import Optional

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


def calmetric(pred_recon, gt_recon):
    if gt_recon.ndim == 4:
        psnr_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        ssim_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        nmse_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-2]):
            for j in range(gt_recon.shape[-1]):
                pred, gt = pred_recon[:, :, i, j], gt_recon[:, :, i, j]
                psnr_array[i, j] = psnr(gt / gt.max(), pred / pred.max())
                ssim_array[i, j] = ssim(gt / gt.max(), pred / pred.max())
                nmse_array[i, j] = nmse(gt / gt.max(), pred / pred.max())
    elif gt_recon.ndim == 2:
        # psnr_array = np.zeros(1)
        # ssim_array = np.zeros(1)
        # nmse_array = np.zeros(1)
        psnr_array = psnr(gt_recon / gt_recon.max(), pred_recon / pred_recon.max())
        ssim_array = ssim(gt_recon / gt_recon.max(), pred_recon / pred_recon.max())
        nmse_array = nmse(gt_recon / gt_recon.max(), pred_recon / pred_recon.max())
    else:
        psnr_array = np.zeros((1, gt_recon.shape[-1]))
        ssim_array = np.zeros((1, gt_recon.shape[-1]))
        nmse_array = np.zeros((1, gt_recon.shape[-1]))

        for j in range(gt_recon.shape[-1]):
            pred, gt = pred_recon[:, :, j], gt_recon[:, :, j]
            psnr_array[0,j] = psnr(gt / gt.max(), pred / pred.max())
            ssim_array[0,j] = ssim(gt / gt.max(), pred / pred.max())
            nmse_array[0,j] = nmse(gt / gt.max(), pred / pred.max())

    return psnr_array, ssim_array, nmse_array

