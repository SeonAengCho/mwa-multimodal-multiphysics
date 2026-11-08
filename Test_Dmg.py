import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

from config import dmg_model, data_dir, weight_dir, result_dir, device, test_seed
from utils import ImagePlot
from loss import diceloss, iouloss, hausdorff, asd

from model import build_model, get_model_name


class DamageTestDataset(Dataset):
    """Test samples of one split.

    Each item: image (2, 81, 81, 81) = [tumor, vessel], coord (6,), cond (2,), label (81, 81, 81) binary damage.
    """
    def __init__(self, split):
        self.tumoronly = np.load(f"{data_dir}/{split}_input_onlytumor.npy", mmap_mode="r")
        self.vessel = np.load(f"{data_dir}/{split}_input_vessel.npy", mmap_mode="r")
        self.coord = np.load(f"{data_dir}/{split}_input_needle.npy", mmap_mode="r")
        self.cond = np.load(f"{data_dir}/{split}_input_cond.npy", mmap_mode="r")
        self.label = np.load(f"{data_dir}/{split}_label_dmg.npy", mmap_mode="r")
        if self.tumoronly.shape != self.vessel.shape:
            raise ValueError(f"{split}: tumoronly and vessel shapes must match")
        if len(self.coord) != len(self.label) or len(self.cond) != len(self.label):
            raise ValueError(f"{split}: input and label sample counts must match")
        if len(self.tumoronly) == 0 or len(self.label) % len(self.tumoronly):
            raise ValueError(f"{split}: samples must be evenly grouped by tumor")
        if self.tumoronly.shape[1:] != self.label.shape[1:]:
            raise ValueError(f"{split}: image and label spatial shapes must match")
        self.samples_per_tumor = len(self.label) // len(self.tumoronly)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        tumor_index = index // self.samples_per_tumor
        tumoronly = torch.tensor(self.tumoronly[tumor_index], dtype=torch.float32)
        vessel = torch.tensor(self.vessel[tumor_index], dtype=torch.float32)
        # stacked here because ImagePlot expects (image, coord, cond, label) items
        image = torch.stack((tumoronly, vessel), dim=0)
        coord = torch.tensor(self.coord[index], dtype=torch.float32)
        cond = torch.tensor(self.cond[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)
        return image, coord, cond, label


def DataLoad():
    """Foreseen: the last 10% of each training tumor's samples (not used for training).
    Unforeseen: 1,080 random samples of the test tumors (test_seed in config.py).
    """
    print("Data Loading ... ")
    ds_train = DamageTestDataset("train")
    ds_test = DamageTestDataset("test")
    print("Preparing Dataset ... ")

    seed = test_seed["dmg"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_indices = torch.tensor([
        i * ds_train.samples_per_tumor + j
        for i in range(len(ds_train.tumoronly))
        for j in range(int(ds_train.samples_per_tumor * 0.9), ds_train.samples_per_tumor)
    ])
    ds_train = Subset(ds_train, train_indices)

    subset_indices = torch.randperm(len(ds_test))[:1080]
    ds_test = Subset(ds_test, subset_indices)

    loader_foreseen = DataLoader(ds_train, batch_size=1, shuffle=False)
    loader_unforeseen = DataLoader(ds_test, batch_size=1, shuffle=False)
    return loader_foreseen, loader_unforeseen


def Synchronize(device):
    """Wait for queued GPU work so that time.time() measures the forward pass only."""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def TestModel(model, device, criterion, loader_test, th, mode):
    """Print Dice, IoU, HD95, ASD (prediction thresholded at th) and inference time; return the predictions."""
    # (name, fn(label, binary prediction)); every metric is computed per sample, then averaged
    metrics = [
        ("Dice", lambda t, p: 1 - criterion(t, p)),
        ("IoU", lambda t, p: iouloss(t, p)),
        ("HD95", lambda t, p: hausdorff(t, p, spacing=(1, 1, 1), hd95=True)),
        # Paper Eq. (16): ASD(X, Y) with X = ground truth, Y = prediction (GT -> prediction).
        ("ASD", lambda t, p: asd(t, p, spacing=(1, 1, 1), direction="gt2pr")),
    ]
    metric_sum = {name: 0.0 for name, _ in metrics}
    metric_count = {name: 0 for name, _ in metrics}
    output_list = []
    infer_time = 0.0

    model.eval()
    with torch.no_grad():
        with tqdm(loader_test) as tepoch:
            for image, coord, cond, label in tepoch:
                image = image.to(device)
                coord = coord.to(device)
                cond = cond.to(device)
                label = label.to(device)

                # Time only the forward pass (metric computation is excluded).
                Synchronize(device)
                start_time = time.time()
                outputs = model(image, coord, cond)
                Synchronize(device)
                infer_time += time.time() - start_time

                outputs = (outputs > th).float()
                output_list.append(outputs.detach().cpu().numpy())

                for name, fn in metrics:
                    value = float(fn(label, outputs))
                    # HD95/ASD are undefined (NaN) when only one of the masks is empty.
                    if np.isfinite(value):
                        metric_sum[name] += value
                        metric_count[name] += 1

    n = len(loader_test)
    for name, _ in metrics:
        count = metric_count[name]
        mean = metric_sum[name] / count if count else float("nan")
        skipped = f"  ({n - count} undefined samples skipped)" if count < n else ""
        print(f"{mode} {name} : {mean:.4f}{skipped}")
    print(f"Inference time : {infer_time / n:.4f}sec")

    return output_list


if __name__ == "__main__":
    # The model is selected in config.py (dmg_model), the same one used for training.
    model = build_model(dmg_model, mode="dmg")

    model_name = get_model_name(model)
    model = model.to(device)
    weight_path = Path(weight_dir) / f"{model_name}.pth"
    print(f"Loading model weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    loader_foreseen, loader_unforeseen = DataLoad()
    image_foreseen = TestModel(model, device, diceloss, loader_foreseen, 0.5, "Foreseen")
    image_unforeseen = TestModel(model, device, diceloss, loader_unforeseen, 0.5, "Unforeseen")

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    ImagePlot(loader_foreseen, image_foreseen, sample_idx=16)
    ImagePlot(loader_unforeseen, image_unforeseen, sample_idx=16)
    ImagePlot(loader_unforeseen, image_unforeseen, sample_idx=26)
    ImagePlot(loader_unforeseen, image_unforeseen, sample_idx=36)
    ImagePlot(loader_unforeseen, image_unforeseen, sample_idx=46)
