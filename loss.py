import torch
import torch.nn.functional as F
import numpy as np

from scipy.ndimage import binary_erosion, distance_transform_edt


# ---- Temperature ----

def temploss(pred, target):
    """Training loss of the temperature model: 0.5 * MSE + 0.5 * gradient difference loss."""
    mse = F.mse_loss(pred, target)
    gdl = compute_gdl(pred, target)

    return 0.5 * mse + 0.5 * gdl


def compute_gdl(pred, target, method='mse'):
    """Gradient difference loss: error between the spatial differences of pred and target along each axis."""
    p_dx = torch.diff(pred, dim=-1)
    t_dx = torch.diff(target, dim=-1)

    p_dy = torch.diff(pred, dim=-2)
    t_dy = torch.diff(target, dim=-2)

    p_dz = torch.diff(pred, dim=-3)
    t_dz = torch.diff(target, dim=-3)

    if method == 'mse':
        loss_x = F.mse_loss(p_dx, t_dx)
        loss_y = F.mse_loss(p_dy, t_dy)
        loss_z = F.mse_loss(p_dz, t_dz)
    else:  # l1
        loss_x = F.l1_loss(p_dx, t_dx)
        loss_y = F.l1_loss(p_dy, t_dy)
        loss_z = F.l1_loss(p_dz, t_dz)

    return loss_x + loss_y + loss_z


def roi_dice(pred, target, roi_th=50):
    """Dice of the region above roi_th degrees C (pred and target in degrees C)."""
    target = (target > roi_th).float()
    pred = (pred > roi_th).float()

    return 1 - diceloss(target, pred)


# ---- Thermal damage ----

def dmgloss(target, sdf, pred, alpha=0.5, beta=0.5):
    """Training loss of the damage model: alpha * Dice loss + beta * boundary loss.

    The boundary loss weights the prediction by |signed distance| to the label boundary (sdf).
    """
    d_loss = diceloss(target, pred)
    b_loss = torch.mean(pred * torch.abs(sdf))
    loss = alpha * d_loss + beta * b_loss

    return loss


def diceloss(target, pred):
    """1 - Dice, with Dice computed per sample and averaged over the batch. Use 1 - diceloss() for Dice."""
    smooth = 1.
    iflat = pred.contiguous().view(pred.shape[0], -1)
    tflat = target.contiguous().view(target.shape[0], -1)
    intersection = (iflat * tflat).sum(1)
    A_sum = torch.sum(iflat * iflat, 1)
    B_sum = torch.sum(tflat * tflat, 1)

    return (1 - (2 * (intersection + smooth) / (A_sum + B_sum + smooth))).mean()


def iouloss(target, pred, smooth=1.0):
    """IoU score (not 1 - IoU), computed over the whole batch."""
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    A_sum = (pred * pred).sum()
    B_sum = (target * target).sum()
    union = A_sum + B_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def hausdorff(gt_bhwd, pr_bhwd, spacing=(1,1,1), hd95=False):
    """Hausdorff distance (or HD95 if hd95=True) between the mask surfaces, averaged over the batch.

    gt_bhwd, pr_bhwd: binary masks (B, 81, 81, 81). Samples where only one mask is empty are skipped (NaN).
    """
    if hasattr(gt_bhwd, "detach"): gt_bhwd = gt_bhwd.detach().cpu().numpy()
    if hasattr(pr_bhwd, "detach"): pr_bhwd = pr_bhwd.detach().cpu().numpy()
    gt_bhwd = gt_bhwd.astype(bool)
    pr_bhwd = pr_bhwd.astype(bool)

    vals = []
    for gt, pr in zip(gt_bhwd, pr_bhwd):
        if gt.sum() == 0 and pr.sum() == 0:
            vals.append(0.0); continue
        if gt.sum() == 0 or pr.sum() == 0:
            vals.append(np.nan); continue

        # surface voxels
        s_gt = gt & ~binary_erosion(gt)
        s_pr = pr & ~binary_erosion(pr)

        # distance from every voxel to the nearest surface voxel
        dt_pr = distance_transform_edt(~s_pr, sampling=spacing)
        dt_gt = distance_transform_edt(~s_gt, sampling=spacing)

        d_gt2pr = dt_pr[s_gt]
        d_pr2gt = dt_gt[s_pr]

        if hd95:
            all_d = np.concatenate([d_gt2pr, d_pr2gt])
            vals.append(np.percentile(all_d, 95))
        else:
            vals.append(max(d_gt2pr.max(), d_pr2gt.max()))

    # NaN (not a warning) when every sample is undefined
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def asd(gt_bhwd, pr_bhwd, spacing=(1,1,1), direction="symmetric"):
    """Average surface distance, averaged over the batch.

    direction:
      "gt2pr":     mean distance from each GT surface voxel to the predicted surface
                   (paper Eq. (16) with X = ground truth, Y = prediction)
      "pr2gt":     mean distance from each predicted surface voxel to the GT surface
      "symmetric": average of the two
    Samples where only one mask is empty are skipped (NaN).
    """
    if direction not in ("symmetric", "gt2pr", "pr2gt"):
        raise ValueError(f"unknown direction: {direction}")

    if hasattr(gt_bhwd, "detach"): gt_bhwd = gt_bhwd.detach().cpu().numpy()
    if hasattr(pr_bhwd, "detach"): pr_bhwd = pr_bhwd.detach().cpu().numpy()
    gt_bhwd = gt_bhwd.astype(bool)
    pr_bhwd = pr_bhwd.astype(bool)

    vals = []
    for gt, pr in zip(gt_bhwd, pr_bhwd):
        if gt.sum() == 0 and pr.sum() == 0:
            vals.append(0.0); continue
        if gt.sum() == 0 or pr.sum() == 0:
            vals.append(np.nan); continue

        # surface voxels
        s_gt = gt & ~binary_erosion(gt)
        s_pr = pr & ~binary_erosion(pr)

        # distance from every voxel to the nearest surface voxel
        dt_pr = distance_transform_edt(~s_pr, sampling=spacing)
        dt_gt = distance_transform_edt(~s_gt, sampling=spacing)

        gt2pr = dt_pr[s_gt].mean()
        pr2gt = dt_gt[s_pr].mean()

        if direction == "gt2pr":
            vals.append(gt2pr)
        elif direction == "pr2gt":
            vals.append(pr2gt)
        else:
            vals.append(0.5 * (gt2pr + pr2gt))

    # NaN (not a warning) when every sample is undefined
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
