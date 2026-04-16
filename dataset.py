import os
import glob
import json
import hashlib
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoTokenizer

# 1. Scenario Mapping Dictionary
SCENARIO_FOLDER_MAP = {
    'T01': '01.Grassland', 'T02': '02.Island', 'T03': '03.Ocean',
    'T04': '04.Lake', 'T05': '05.Suburban', 'T06': '06.DenseUrban',
    'T07': '07.Rural', 'T08': '08.OrdinaryUrban', 'T09': '09.Desert',
    'T10': '10.Mountainous', 'T11': '11.Forest'
}
SCENARIO_ID_MAP = {k: i for i, k in enumerate(sorted(SCENARIO_FOLDER_MAP.keys()))}
F_IDX_TO_CODE = {0: 'f00', 1: 'f01', 2: 'f02', 3: 'f03', 4: 'f04'}


def parse_tx_info(txt_path):
    tx_dict = {}
    if not os.path.exists(txt_path):
        print(f"[WARNING] Transmitter file not found: {txt_path}")
        return {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        key = parts[0]
        try:
            tx_dict[key] = json.loads(parts[1])
        except:
            continue
    return tx_dict

# 2. Core Dataset Class

class SpectrumNetDataset(Dataset):
    def __init__(self, root_dir, phase="train", aligned=False):
        self.root_dir = root_dir
        self.phase = phase
        self.aligned = aligned

        self.dir_png = os.path.join(root_dir, 'png')
        self.dir_npz = os.path.join(root_dir, 'npz')
        self.tx_info_path = os.path.join(root_dir, 'tx_info.txt')

        self.tx_raw_data = parse_tx_info(self.tx_info_path)
        self.samples = self._scan_dataset()

        self.num_patches = 16
        self.patch_size = 32
        self.grid_size = 4
        self.current_epoch = 0

        pec_path = os.path.join(self.root_dir, "vlm_semantics.json")
        if not os.path.exists(pec_path):
            raise FileNotFoundError(f"[ERROR] Semantic dictionary not found: {pec_path}")

        self.pec_dict = {}
        try:
            with open(pec_path, 'r', encoding='utf-8') as f:
                self.pec_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(pec_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if '": "' in line:
                        parts = line.split('": "', 1)
                        k = parts[0].strip().strip('"{}, ')
                        v = parts[1].strip().strip('"{}, ')
                        if k and v: self.pec_dict[k] = v
                    elif '"' in line and not line.startswith('{') and not line.startswith('}'):
                        first_quote = line.find('"')
                        if first_quote != -1:
                            k = line[:first_quote].strip().strip('"{}, :')
                            v = line[first_quote:].strip().strip('"{}, ')
                            if k and v: self.pec_dict[k] = v

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _scan_dataset(self):
        npz_files = glob.glob(os.path.join(self.dir_npz, "**", "*_bdtr.npz"), recursive=True)
        npz_index = {os.path.basename(p).replace("_bdtr.npz", ""): p for p in npz_files}

        png_files = glob.glob(os.path.join(self.dir_png, "**", "*.png"), recursive=True)
        groups = defaultdict(list)

        for png_path in png_files:
            filename = os.path.basename(png_path)
            if '_f' not in filename: continue
            core_id = filename.split('_f')[0]
            if core_id not in npz_index: continue
            if core_id not in self.tx_raw_data: continue

            try:
                z_loc = filename.find('_z')
                z_str = filename[z_loc + 2: z_loc + 4]
                if not z_str.isdigit(): z_str = filename[z_loc + 2]
                if int(z_str) != 0: continue

                f_loc = filename.find('_f')
                f_idx = int(filename[f_loc + 2: f_loc + 4])
                scenario_str = core_id[:3]
                scenario_idx = SCENARIO_ID_MAP.get(scenario_str, 0)
                d_pos = core_id.find('D')
                map_num = int(core_id[d_pos + 1: d_pos + 5])
                c_pos = core_id.find('C')
                climate_str = core_id[c_pos:d_pos]
            except:
                continue

            is_valid_split = False
            is_short_data = (scenario_str == 'T10' and climate_str == 'C1')

            if is_short_data:
                if self.phase == "train" and map_num <= 5:
                    is_valid_split = True
                elif self.phase == "val" and map_num == 6:
                    is_valid_split = True
                elif self.phase == "test" and map_num == 7:
                    is_valid_split = True
            else:
                remainder = map_num % 10
                if self.phase == "train" and remainder <= 7:
                    is_valid_split = True
                elif self.phase == "val" and remainder == 8:
                    is_valid_split = True
                elif self.phase == "test" and remainder == 9:
                    is_valid_split = True

            if not is_valid_split: continue

            groups[core_id].append({
                'input_path': npz_index[core_id],
                'tx_key': core_id,
                'label_path': png_path,
                'scenario_idx': scenario_idx,
                'f_idx': f_idx
            })

        valid_samples = []
        for core_id, sample_list in groups.items():
            if self.aligned:
                if len(set(s['f_idx'] for s in sample_list)) == 5:
                    valid_samples.extend(sample_list)
            else:
                valid_samples.extend(sample_list)
        return valid_samples

    def __len__(self):
        return len(self.samples) * self.num_patches

    def __getitem__(self, idx):
        sample_idx = idx // self.num_patches
        patch_idx = idx % self.num_patches
        info = self.samples[sample_idx]
 # Physical Topology Loading
        try:
            npz = np.load(info['input_path'])
            bldg_map = npz['inBldg_zyx'][0].astype(np.float32)[np.newaxis, :, :]
        except:
            bldg_map = np.zeros((1, 128, 128), dtype=np.float32)

        tx_list = self.tx_raw_data.get(info['tx_key'], [])
        tx_map = np.zeros((1, 128, 128), dtype=np.float32)
        if len(tx_list) > 0:
            tx = tx_list[0]
            tx_x_norm, tx_y_norm = (tx['x'] / 64.0) - 1.0, (tx['y'] / 64.0) - 1.0
            x_idx, y_idx = min(max(int(round(tx['x'])), 0), 127), min(max(int(round(tx['y'])), 0), 127)
            tx_map[0, y_idx, x_idx] = 1.0
        else:
            tx_x_norm, tx_y_norm = 0.0, 0.0
        tx_pos_tensor = torch.tensor([tx_x_norm, tx_y_norm], dtype=torch.float32)

        combined_map = bldg_map.copy()
        combined_map[tx_map == 1.0] = 2.0

        try:
            img = Image.open(info['label_path']).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
            label_tensor = (np.array(img).astype(np.float32) / 255.0)[np.newaxis, :, :]
        except:
            label_tensor = np.zeros((1, 128, 128), dtype=np.float32)
# Deterministic Reproducibility Locks for Sparse Masks
        if self.phase == 'train':
            # Hash seed combined with epoch ensures dynamic data augmentation
            # while maintaining exact reproducibility across different runs.
            seed_str = f"{info['tx_key']}_epoch_{self.current_epoch}"
            seed_val = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
            rng = np.random.RandomState(seed_val)
            sparse_mask = (rng.rand(1, 128, 128) < 0.01).astype(np.float32)
        else:
            # Fixed hash seed for validation/testing to ensure consistent evaluation.
            seed_str = info['tx_key']
            seed_val = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
            rng = np.random.RandomState(seed_val)
            sparse_mask = (rng.rand(1, 128, 128) < 0.01).astype(np.float32)

        sparse_map = label_tensor * sparse_mask
        depth_path = info['label_path'].replace('/png/', '/depth_data_v2/').replace('.png', '.npy')
        try:
            depth_data = np.clip(np.load(depth_path).astype(np.float32) / 200.0, 0.0, 1.0)
            depth_tensor = depth_data[np.newaxis, :, :]
        except:
            depth_tensor = np.zeros((1, 128, 128), dtype=np.float32)

        input_tensor_full = np.concatenate([combined_map, sparse_map, depth_tensor], axis=0)
        label_tensor_full = label_tensor

        # ----------------------------------------------------
        # Local Patch Cropping
        # ----------------------------------------------------
        row_idx, col_idx = patch_idx // self.grid_size, patch_idx % self.grid_size
        row_start, col_start = row_idx * self.patch_size, col_idx * self.patch_size

        input_patch = input_tensor_full[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
        label_patch = label_tensor_full[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]

        global_coords = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                global_y = row_start + i
                global_x = col_start + j
                global_coords[0, i, j] = (global_x / 64.0) - 1.0
                global_coords[1, i, j] = (global_y / 64.0) - 1.0
        global_coords_tensor = torch.from_numpy(global_coords)

        # ----------------------------------------------------
        # Global Semantic Context
        # ----------------------------------------------------
        f_code = F_IDX_TO_CODE.get(info['f_idx'], 'f00')
        dict_key = f"{info['tx_key']}_{f_code}"

        pec_text = self.pec_dict.get(dict_key, "Standard radio propagation environment.")

        encoded = self.tokenizer(
            pec_text,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return (
            torch.from_numpy(input_patch).float(),
            torch.from_numpy(label_patch).float(),
            torch.tensor(info['f_idx']).long(),
            torch.tensor(info['scenario_idx']).long(),
            tx_pos_tensor,
            input_ids,
            attention_mask,
            global_coords_tensor
        )