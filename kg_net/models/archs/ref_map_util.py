import torch
import torch.nn.functional as F
import pdb


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.
    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.

    Returns:
        patches (Tensor): patches, shape: (c, patch_size,
            patch_size, n_patches).
    """

    c, h, w = inputs.shape
    patches = inputs.unfold(1, patch_size, stride)\
                    .unfold(2, patch_size, stride)\
                    .reshape(c, -1, patch_size, patch_size)\
                    .permute(0, 2, 3, 1)
    return patches


def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):
    """Patch matching between input and reference features.
    Args:

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)   # [64, 3, 3, 1444]
    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape  # [64, 40, 40]
    batch_size = int(1024.**2 * 512 / (h * w))  # 335544
    n_patches = patches_ref.shape[-1]   # 1444

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)  # [64, 3, 3, 1444]
        corr = F.conv2d(feat_input.unsqueeze(0),
                        batch.permute(3, 0, 1, 2),   # [1444, 64, 3, 3]
                        stride=input_stride)         # [1, 1444, 38, 38]

        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0) # [38, 38], [38, 38]

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp       # [38, 38], [38, 38]
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val


def topk_feature_match_index(feat_input,
                             feat_ref,
                             patch_size=3,
                             input_stride=1,
                             ref_stride=1,
                             is_norm=True,
                             norm_input=False,
                             K=2):
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)   # [64, 3, 3, 1444]
    #
    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape  # [64, 40, 40]
    batch_size = int(1024.**2 * 512 / (h * w))  # 335544
    n_patches = patches_ref.shape[-1]   # 1444

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)  # [64, 3, 3, 1444]
        corr = F.conv2d(feat_input.unsqueeze(0),
                        batch.permute(3, 0, 1, 2),   # [1444, 64, 3, 3]
                        stride=input_stride)         # [1, 1444, 38, 38]

        max_val_tmp, max_idx_tmp = torch.topk(corr.squeeze(0), K, 0) # [K, 38, 38], [K, 38, 38]

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp       # [K, 38, 38], [K, 38, 38]
        else:
            for k in range(K):
                indices = max_val_tmp[k] > max_val[k]
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val
