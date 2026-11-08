import os
import numpy as np
import matplotlib.pyplot as plt
from config import result_dir


def Scaling(dataset):
    """Temperature in degrees C -> network scale: 37 C -> 0, 100 C -> 1."""
    return (dataset - 37) / 63


def InverseScaling(dataset):
    """Network scale -> temperature in degrees C (inverse of Scaling)."""
    return (dataset * 63) + 37


def ImagePlot(dataloader, model_output, slice_idx=None, sample_idx=0):
    """Save the label and prediction of one test sample side by side to result_dir.

    The slice is taken at the needle's first coordinate. model_output is the list returned by TestModel.
    """
    batch = dataloader.dataset[sample_idx]
    coord = batch[1].detach().cpu().numpy()
    cond = batch[2].detach().cpu().numpy()
    label_volume = batch[3].detach().cpu().numpy()                   # (81, 81, 81)
    output_volume = np.squeeze(model_output[sample_idx], axis=(0))   # (81, 81, 81)

    # undo the [0, 1] normalization: needle position 17-62 voxels, time 5-29, power 45-100 W
    coord_denorm = coord[:3] * (62 - 17) + 17
    time_denorm = int(cond[0] * (29 - 5) + 5)
    watt_denorm = int(cond[1] * (100 - 45) + 45)
    slice_idx = int(coord_denorm[0])

    print(slice_idx)
    label_slice = label_volume[slice_idx]
    output_slice = output_volume[slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(label_slice, cmap='viridis')
    axes[0].set_title(f"Label Slice [{slice_idx}]")
    axes[1].imshow(output_slice, cmap='viridis')
    axes[1].set_title(f"Output Slice [{slice_idx}]")
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Coord = {coord_denorm}, watt = {watt_denorm}, time = {time_denorm}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(result_dir, f"result_sample{sample_idx}_slice{slice_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    plt.show()
