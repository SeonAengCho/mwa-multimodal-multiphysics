import torch
import numpy as np
import time
import copy
from pathlib import Path
from config import (dmg_model, data_dir, weight_dir, learning_rate, epoch_num,
                    batch_size, device, train_seed, valid_seed)
from tqdm import tqdm
from loss import dmgloss, diceloss
from torch.utils.data import Dataset, DataLoader, Subset

from model import build_model, get_model_name, set_train_seed, pick_seed

from scipy.ndimage import distance_transform_edt as edt


def compute_sdf(mask_np):
    """Signed distance to the mask boundary (outside +, inside -), squashed with tanh: (81, 81, 81)."""
    mask = mask_np.astype(bool)
    pos = mask                  # inside region
    neg = ~mask                 # outside region

    dist_out = edt(neg)         # distance from outside to nearest inside
    dist_in = edt(pos)          # distance from inside to nearest outside

    sdf = dist_out - dist_in

    normalized_sdf = np.tanh(sdf)

    return normalized_sdf.astype(np.float32)


class DamageDataset(Dataset):
    """Damage samples of one split.

    Each item: tumoronly (81, 81, 81), vessel (81, 81, 81), coord (6,), cond (2,), label (81, 81, 81),
    and sdf (81, 81, 81) if include_sdf.
    """
    def __init__(self, split, include_sdf=False):
        self.tumoronly = np.load(f"{data_dir}/{split}_input_onlytumor.npy", mmap_mode="r")
        self.vessel = np.load(f"{data_dir}/{split}_input_vessel.npy", mmap_mode="r")
        self.coord = np.load(f"{data_dir}/{split}_input_needle.npy", mmap_mode="r")
        self.cond = np.load(f"{data_dir}/{split}_input_cond.npy", mmap_mode="r")
        self.label = np.load(f"{data_dir}/{split}_label_dmg.npy", mmap_mode="r")
        self.include_sdf = include_sdf
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
        # tumor/vessel volumes are stored once per tumor and shared by its samples
        sample = tuple(torch.tensor(array, dtype=torch.float32) for array in (
            self.tumoronly[tumor_index], self.vessel[tumor_index],
            self.coord[index], self.cond[index], self.label[index]
        ))
        if self.include_sdf:
            sample += (torch.from_numpy(compute_sdf(self.label[index])),)
        return sample


def DataLoad(batch):
    """Train loader: first 90% of each training tumor's samples (the last 10% are the Foreseen test set).
    Validation loader: 1,080 random validation samples (valid_seed in config.py), used to pick the best epoch.
    """
    print ("Data Loading ... ")

    ds_train = DamageDataset("train", include_sdf=True)
    ds_valid = DamageDataset("valid")

    print ("Preparing Dataset ... ")

    train_indices = torch.tensor([
    i * ds_train.samples_per_tumor + j
    for i in range(len(ds_train.tumoronly))
    for j in range(int(ds_train.samples_per_tumor * 0.9))])
    ds_train = Subset(ds_train, train_indices)

    subset_indices = torch.randperm(len(ds_valid), generator=torch.Generator().manual_seed(pick_seed(valid_seed, "valid_seed")))[:1080]
    ds_valid = Subset(ds_valid, subset_indices)

    loader_train = DataLoader(ds_train, batch_size = batch, shuffle = True, num_workers = 8)
    loader_valid = DataLoader(ds_valid, batch_size = batch, shuffle = True, num_workers = 8)

    return loader_train, loader_valid


def TrainModel(model, learning_rate, epoch_num, device, criterion, loader_train, loader_valid, patience=15):
    """Adam with lr * 0.98^epoch. Keeps the weights of the epoch with the lowest validation loss and
    stops early after `patience` epochs without improvement.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    learning_schedule = lambda epoch: 0.98 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_schedule)

    train_loss_list = []
    valid_loss_list = []

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epoch_num):
        train_loss = 0
        valid_loss = 0

        model.train()
        with tqdm(loader_train, desc=f"Epoch {epoch+1}/{epoch_num}") as tepoch:
            start_time = time.time()

            for tumoronly, vessel, coord, cond, label, sdf in tepoch:
                image = torch.stack((tumoronly, vessel), dim=1).to(device)  # (B, 2, 81, 81, 81)
                coord = coord.to(device)
                cond = cond.to(device)
                label = label.to(device)
                sdf = sdf.to(device)

                outputs = model(image, coord, cond)
                loss = criterion(label, sdf, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            end_time = time.time()
            execution_time = end_time - start_time
            train_loss_mean = train_loss / len(loader_train)
            print(f"{epoch+1} epoch Train Loss : {train_loss_mean:.4f}")
            print("Execution time:", execution_time, "seconds")
            train_loss_list.append(train_loss_mean)

        # validation: Dice loss of the thresholded prediction
        model.eval()
        with torch.no_grad():
            for tumoronly, vessel, coord, cond, label in loader_valid:
                image = torch.stack((tumoronly, vessel), dim=1).to(device)  # (B, 2, 81, 81, 81)
                coord = coord.to(device)
                cond = cond.to(device)
                label = label.to(device)
                outputs = model(image, coord, cond)
                outputs = (outputs > 0.5).float()
                loss = diceloss(label, outputs)
                valid_loss += loss.item() * len(label)

            valid_loss_mean = valid_loss / len(loader_valid.dataset)
            print(f"{epoch+1} Validation Loss : {valid_loss_mean:.4f}")
            valid_loss_list.append(valid_loss_mean)

        scheduler.step()

        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            print("Validation loss improved. Model weights updated.\n")
        else:
            counter += 1
            print(f"No improvement in validation loss. Counter: {counter}/{patience}\n")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return train_loss_list, valid_loss_list


if __name__ == "__main__":
    # The model is selected in config.py (dmg_model).
    set_train_seed(train_seed)
    model = build_model(dmg_model, mode="dmg")

    model_name = get_model_name(model)
    model = model.to(device)
    print(f"training mode is {model_name}")

    loader_train, loader_valid = DataLoad(batch_size)
    train_loss_list, valid_loss_list = TrainModel(
        model, learning_rate, epoch_num, device, dmgloss, loader_train, loader_valid
    )

    Path(weight_dir).mkdir(parents=True, exist_ok=True)
    weight_path = Path(weight_dir) / f"{model_name}.pth"
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved to {weight_path}")
