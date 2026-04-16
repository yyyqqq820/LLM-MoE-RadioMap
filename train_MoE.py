import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import SpectrumNetDataset
from module import MoESpectrumNet


# Hyperparameter Configuration

BATCH_SIZE = 1024
VAL_BATCH_SIZE = 16
EPOCHS_STAGE2 = 50
LR_STAGE2 = 2e-4
EPOCHS_STAGE3 = 20
LR_STAGE3 = 1e-5

USE_STRICT_MODE = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "SpectrumNet"))

STAGE1_CKPT_DIR = os.path.join(CURRENT_DIR, "Checkpoints_Stage1_MSE")
SAVE_DIR = os.path.join(CURRENT_DIR, "Checkpoints_MoE_TwoStage")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


sys.stdout = Logger(os.path.join(CURRENT_DIR, "Log_MoE_Training.txt"))


# Reproducibility Locks

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_grid_wise_metrics(preds, targets):
    grid_mse = torch.mean((preds - targets) ** 2).item()
    grid_rmse = np.sqrt(grid_mse)
    target_power = torch.mean(targets ** 2).item()
    grid_nmse = grid_mse / (target_power + 1e-10)
    grid_psnr = 10 * np.log10(1.0 / (grid_mse + 1e-10))
    return grid_mse, grid_rmse, grid_nmse, grid_psnr, 1


def load_expert_weights(model):
    expert_to_file = {
        0: "best_ResCNN_MSE.pth",
        1: "best_ViT_MSE.pth",
        2: "best_DNN_MSE.pth",
        3: "best_GNN_GridRefined.pth"
    }

    for i, filename in expert_to_file.items():
        pth_path = os.path.join(STAGE1_CKPT_DIR, filename)
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Expert weights not found: {pth_path}")

        checkpoint = torch.load(pth_path, map_location=DEVICE)
        current_dict = model.experts[i].state_dict()
        load_dict = {k: v for k, v in checkpoint.items() if k in current_dict and v.shape == current_dict[k].shape}
        model.experts[i].load_state_dict(load_dict, strict=False)


# Main Training Loop

def main():
    seed_everything(42)

    train_ds = SpectrumNetDataset(DATA_ROOT, "train", aligned=USE_STRICT_MODE)
    val_ds = SpectrumNetDataset(DATA_ROOT, "val", aligned=USE_STRICT_MODE)

    g = torch.Generator()
    g.manual_seed(42)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )
    val_dl = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=2, drop_last=False,
        worker_init_fn=seed_worker, generator=g
    )

    model = MoESpectrumNet(n_channels=3, n_classes=1, hidden_dim=64).to(DEVICE)
    load_expert_weights(model)
    criterion = nn.MSELoss()

    best_overall_val_mse = float('inf')

    # Stage 2: Gating Network Warm-up
    for param in model.experts.parameters(): param.requires_grad = False
    optimizer_s2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2)
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=EPOCHS_STAGE2)

    for epoch in range(EPOCHS_STAGE2):
        train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        train_act_accum = torch.zeros(4).to(DEVICE)

        for d, l, _, _, tx_pos, input_ids, att_mask, g_coords in train_dl:
            inputs, labels = d.to(DEVICE), l.to(DEVICE)
            tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)
            input_ids, att_mask = input_ids.to(DEVICE), att_mask.to(DEVICE)

            preds, weights, aux_loss = model(inputs, input_ids, att_mask, tx_pos, g_coords)
            main_loss = criterion(preds, labels)

            aux_weight = 0.05 if epoch < 10 else 0.0
            loss = main_loss + aux_weight * aux_loss

            optimizer_s2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer_s2.step()

            running_loss += main_loss.item()
            train_act_accum += weights.detach().sum(dim=0)

        model.eval()
        sum_mse, sum_rmse, sum_nmse, sum_psnr, total_grids = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for d, l, _, _, tx_pos, input_ids, att_mask, g_coords in val_dl:
                if d.shape[0] != VAL_BATCH_SIZE: continue
                inputs, labels = d.to(DEVICE), l.to(DEVICE)
                tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)
                input_ids, att_mask = input_ids.to(DEVICE), att_mask.to(DEVICE)

                preds, _, _ = model(inputs, input_ids, att_mask, tx_pos, g_coords)

                g_mse, g_rmse, g_nmse, g_psnr, g_cnt = compute_grid_wise_metrics(preds, labels)
                sum_mse += g_mse;
                sum_rmse += g_rmse;
                sum_nmse += g_nmse;
                sum_psnr += g_psnr;
                total_grids += g_cnt

        train_mse = running_loss / len(train_dl)
        avg_mse, avg_rmse = sum_mse / total_grids, sum_rmse / total_grids
        avg_nmse, avg_psnr = sum_nmse / total_grids, sum_psnr / total_grids
        scheduler_s2.step()

        usage = (train_act_accum / train_act_accum.sum() * 100).cpu().numpy()
        print(
            f"[Stage 2] Ep {epoch + 1:02d} | Train MSE: {train_mse:.5f} | Val MSE: {avg_mse:.5f} | Val NMSE: {avg_nmse:.5f} | Experts -> CNN:{usage[0]:.1f}% ViT:{usage[1]:.1f}% DNN:{usage[2]:.1f}% GNN:{usage[3]:.1f}%")

        if avg_mse < best_overall_val_mse:
            best_overall_val_mse = avg_mse
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "MoE_Best_Model.pth"))

    # Stage 3: Full End-to-End Fine-Tuning
    best_model_path = os.path.join(SAVE_DIR, "MoE_Best_Model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    for param in model.parameters(): param.requires_grad = True
    optimizer_s3 = optim.Adam(model.parameters(), lr=LR_STAGE3, weight_decay=1e-5)
    scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=EPOCHS_STAGE3)

    for epoch in range(EPOCHS_STAGE3):
        train_ds.set_epoch(epoch + EPOCHS_STAGE2)
        model.train()
        running_loss = 0.0
        train_act_accum = torch.zeros(4).to(DEVICE)

        for d, l, _, _, tx_pos, input_ids, att_mask, g_coords in train_dl:
            inputs, labels = d.to(DEVICE), l.to(DEVICE)
            tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)
            input_ids, att_mask = input_ids.to(DEVICE), att_mask.to(DEVICE)

            preds, weights, aux_loss = model(inputs, input_ids, att_mask, tx_pos, g_coords)
            loss = criterion(preds, labels)

            optimizer_s3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_s3.step()

            running_loss += loss.item()
            train_act_accum += weights.detach().sum(dim=0)

        model.eval()
        sum_mse, sum_rmse, sum_nmse, sum_psnr, total_grids = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for d, l, _, _, tx_pos, input_ids, att_mask, g_coords in val_dl:
                if d.shape[0] != VAL_BATCH_SIZE: continue
                inputs, labels = d.to(DEVICE), l.to(DEVICE)
                tx_pos, g_coords = tx_pos.to(DEVICE), g_coords.to(DEVICE)
                input_ids, att_mask = input_ids.to(DEVICE), att_mask.to(DEVICE)

                preds, _, _ = model(inputs, input_ids, att_mask, tx_pos, g_coords)

                g_mse, g_rmse, g_nmse, g_psnr, g_cnt = compute_grid_wise_metrics(preds, labels)
                sum_mse += g_mse;
                sum_rmse += g_rmse;
                sum_nmse += g_nmse;
                sum_psnr += g_psnr;
                total_grids += g_cnt

        train_mse = running_loss / len(train_dl)
        avg_mse, avg_rmse = sum_mse / total_grids, sum_rmse / total_grids
        avg_nmse, avg_psnr = sum_nmse / total_grids, sum_psnr / total_grids
        scheduler_s3.step()

        usage = (train_act_accum / train_act_accum.sum() * 100).cpu().numpy()
        print(
            f"[Stage 3] Ep {epoch + 1:02d} | Train MSE: {train_mse:.5f} | Val MSE: {avg_mse:.5f} | Val NMSE: {avg_nmse:.5f} | Experts -> CNN:{usage[0]:.1f}% ViT:{usage[1]:.1f}% DNN:{usage[2]:.1f}% GNN:{usage[3]:.1f}%")

        if avg_mse < best_overall_val_mse:
            best_overall_val_mse = avg_mse
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "MoE_Best_Model.pth"))


if __name__ == "__main__":
    main()