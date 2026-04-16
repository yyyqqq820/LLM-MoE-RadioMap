import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import SpectrumNetDataset
from module import ExpertResRadioUNet, ExpertTransformer, ExpertDNN, ExpertGNN, RadioResBlock

# Configuration
BATCH_SIZE = 1024
VAL_BATCH_SIZE = 16
LR = 1e-4

EPOCHS_DICT = {
    "ResCNN": 50,
    "ViT": 50,
    "DNN": 50,
    "GNN": 50
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "SpectrumNet"))

SAVE_DIR = os.path.join(CURRENT_DIR, "Checkpoints_Stage1_MSE")
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


sys.stdout = Logger(os.path.join(CURRENT_DIR, "Log_Stage1_AllExperts.txt"))


def compute_grid_wise_metrics(preds, targets):
    grid_mse = torch.mean((preds - targets) ** 2).item()
    grid_rmse = np.sqrt(grid_mse)
    target_power = torch.mean(targets ** 2).item()
    grid_nmse = grid_mse / (target_power + 1e-10)
    grid_psnr = 10 * np.log10(1.0 / (grid_mse + 1e-10))
    return grid_mse, grid_rmse, grid_nmse, grid_psnr, 1


class SingleExpertWrapper(nn.Module):
    def __init__(self, expert_model, expert_name):
        super().__init__()
        self.expert = expert_model
        self.expert_name = expert_name
        self.head = nn.Sequential(
            RadioResBlock(64, 64, kernel=3, padding=1, pool=1),
            RadioResBlock(64, 32, kernel=3, padding=1, pool=1),
            RadioResBlock(32, 16, kernel=3, padding=1, pool=1),
            RadioResBlock(16, 8, kernel=3, padding=1, pool=1),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x, tx_pos=None, g_coords=None):
        if self.expert_name == "DNN":
            out = self.expert(x, tx_pos, g_coords)
        else:
            out = self.expert(x)
        return self.head(out)


# Core Training Logic

def train_single_expert(expert_name, model_class, train_dl, val_dl):
    total_epochs = EPOCHS_DICT.get(expert_name, 50)

    if expert_name == "ResCNN":
        core_expert = model_class(in_channels=3, out_channels=64)
    elif expert_name == "ViT":
        core_expert = model_class(in_channels=3, out_channels=64, patch_size=4, embed_dim=256)
    elif expert_name == "DNN":
        core_expert = model_class(in_channels=3, out_channels=64, base_dim=128)
    elif expert_name == "GNN":
        core_expert = model_class(in_channels=3, out_channels=64, hidden_dim=128, grid_size=32, radius=4.0)

    model = SingleExpertWrapper(core_expert, expert_name).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    best_overall_mse = float('inf')

    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, _, _, tx_pos, _, _, g_coords in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)

            preds = model(inputs, tx_pos, g_coords)
            loss = nn.MSELoss()(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_mse = running_loss / len(train_dl)

        model.eval()
        sum_mse, sum_rmse, sum_nmse, sum_psnr, total_grids = 0.0, 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for inputs, labels, _, _, tx_pos, _, _, g_coords in val_dl:
                if inputs.shape[0] != VAL_BATCH_SIZE:
                    continue
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)

                preds = model(inputs, tx_pos, g_coords)
                g_mse, g_rmse, g_nmse, g_psnr, g_cnt = compute_grid_wise_metrics(preds, labels)
                sum_mse += g_mse;
                sum_rmse += g_rmse;
                sum_nmse += g_nmse;
                sum_psnr += g_psnr;
                total_grids += g_cnt

        avg_val_mse = sum_mse / total_grids
        avg_val_rmse = sum_rmse / total_grids
        avg_val_nmse = sum_nmse / total_grids
        avg_val_psnr = sum_psnr / total_grids
        scheduler.step()

        print(
            f"[{expert_name}] Ep {epoch + 1:02d}/{total_epochs} | Train MSE: {avg_train_mse:.5f} | Val MSE: {avg_val_mse:.5f} | Val NMSE: {avg_val_nmse:.5f} | PSNR: {avg_val_psnr:.2f}dB")

        if avg_val_mse < best_overall_mse:
            best_overall_mse = avg_val_mse
            save_filename = "best_GNN_GridRefined.pth" if expert_name == "GNN" else f"best_{expert_name}_MSE.pth"
            torch.save(model.expert.state_dict(), os.path.join(SAVE_DIR, save_filename))


def main():
    train_ds = SpectrumNetDataset(DATA_ROOT, "train", aligned=True)
    val_ds = SpectrumNetDataset(DATA_ROOT, "val", aligned=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                        drop_last=False)

    EXPERTS = {
        "ResCNN": ExpertResRadioUNet,
        "ViT": ExpertTransformer,
        "DNN": ExpertDNN,
        "GNN": ExpertGNN
    }

    for name, cls in EXPERTS.items():
        train_single_expert(name, cls, train_dl, val_dl)


if __name__ == "__main__":
    main()